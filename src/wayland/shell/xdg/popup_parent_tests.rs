//! Tests for the parent of an `xdg_popup`.
//!
//! A popup is never allowed to be its own ancestor: walking up the parents of a popup has to end
//! at a surface that is not a popup. These tests drive a real `wayland-client` connection against
//! a live [`Display`], so the requests go through the protocol handlers.

use std::{
    os::unix::net::UnixStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use wayland_client::{
    Connection, Dispatch, DispatchError, EventQueue, QueueHandle,
    backend::WaylandError,
    delegate_noop,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{wl_compositor, wl_registry, wl_surface},
};
use wayland_protocols::xdg::shell::client::{
    xdg_popup, xdg_positioner, xdg_surface, xdg_toplevel, xdg_wm_base,
};
use wayland_server::{Client, Display, DisplayHandle, backend::ClientData, protocol::wl_seat::WlSeat};

use crate::{
    utils::Serial,
    wayland::{
        compositor::{CompositorClientState, CompositorHandler, CompositorState},
        shell::xdg::{PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState},
    },
};

// ---------------------------------------------------------------- server

#[derive(Default)]
struct ServerClient {
    compositor: CompositorClientState,
}

impl ClientData for ServerClient {}

struct Server {
    compositor: CompositorState,
    xdg_shell: XdgShellState,
    /// Number of popups the server was told about with `new_popup`.
    popups: usize,
}

impl CompositorHandler for Server {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ServerClient>().unwrap().compositor
    }

    fn commit(&mut self, _surface: &wayland_server::protocol::wl_surface::WlSurface) {}
}

impl XdgShellHandler for Server {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell
    }

    fn new_toplevel(&mut self, _surface: ToplevelSurface) {}

    fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {
        self.popups += 1;
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: WlSeat, _serial: Serial) {}

    fn reposition_request(&mut self, _surface: PopupSurface, _positioner: PositionerState, _token: u32) {}
}

crate::delegate_dispatch2!(Server);

// ---------------------------------------------------------------- client

struct ClientApp;

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for ClientApp {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(ClientApp: wl_compositor::WlCompositor);
delegate_noop!(ClientApp: ignore wl_surface::WlSurface);
delegate_noop!(ClientApp: ignore xdg_wm_base::XdgWmBase);
delegate_noop!(ClientApp: ignore xdg_surface::XdgSurface);
delegate_noop!(ClientApp: ignore xdg_toplevel::XdgToplevel);
delegate_noop!(ClientApp: ignore xdg_popup::XdgPopup);
delegate_noop!(ClientApp: xdg_positioner::XdgPositioner);

// --------------------------------------------------------------- harness

struct Harness {
    server: Arc<Mutex<Server>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    queue: EventQueue<ClientApp>,
    qh: QueueHandle<ClientApp>,
    client: ClientApp,
    compositor: wl_compositor::WlCompositor,
    wm_base: xdg_wm_base::XdgWmBase,
    positioner: xdg_positioner::XdgPositioner,
    /// Client side objects that must stay alive for the whole test.
    surfaces: Vec<wl_surface::WlSurface>,
    xdg_surfaces: Vec<xdg_surface::XdgSurface>,
    toplevels: Vec<xdg_toplevel::XdgToplevel>,
    popups: Vec<xdg_popup::XdgPopup>,
}

impl Harness {
    fn new() -> Self {
        let mut display = Display::<Server>::new().unwrap();
        let mut dh: DisplayHandle = display.handle();
        let server = Arc::new(Mutex::new(Server {
            compositor: CompositorState::new::<Server>(&dh),
            xdg_shell: XdgShellState::new::<Server>(&dh),
            popups: 0,
        }));

        let (client_end, server_end) = UnixStream::pair().unwrap();
        dh.insert_client(server_end, Arc::new(ServerClient::default()))
            .unwrap();

        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let server = server.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    {
                        let mut server = server.lock().unwrap();
                        // A client that causes a protocol error is disconnected; that is not a
                        // failure of the server.
                        let _ = display.dispatch_clients(&mut *server);
                        let _ = display.flush_clients();
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };

        let conn = Connection::from_socket(client_end).unwrap();
        let (globals, queue) = registry_queue_init::<ClientApp>(&conn).unwrap();
        let qh = queue.handle();
        let compositor = globals.bind(&qh, 1..=6, ()).unwrap();
        let wm_base: xdg_wm_base::XdgWmBase = globals.bind(&qh, 1..=6, ()).unwrap();

        let positioner = wm_base.create_positioner(&qh, ());
        positioner.set_size(1, 1);
        positioner.set_anchor_rect(0, 0, 1, 1);

        Harness {
            server,
            stop,
            thread: Some(thread),
            queue,
            qh,
            client: ClientApp,
            compositor,
            wm_base,
            positioner,
            surfaces: Vec::new(),
            xdg_surfaces: Vec::new(),
            toplevels: Vec::new(),
            popups: Vec::new(),
        }
    }

    /// Create a `wl_surface` with an `xdg_surface` and no role; returns its index.
    fn surface(&mut self) -> usize {
        let surface = self.compositor.create_surface(&self.qh, ());
        let xdg_surface = self.wm_base.get_xdg_surface(&surface, &self.qh, ());
        self.surfaces.push(surface);
        self.xdg_surfaces.push(xdg_surface);
        self.surfaces.len() - 1
    }

    fn toplevel(&mut self, surface: usize) {
        let toplevel = self.xdg_surfaces[surface].get_toplevel(&self.qh, ());
        self.toplevels.push(toplevel);
    }

    /// Give `surface` the popup role with `parent` as its parent.
    fn popup(&mut self, surface: usize, parent: usize) {
        let popup = self.xdg_surfaces[surface].get_popup(
            Some(&self.xdg_surfaces[parent]),
            &self.positioner,
            &self.qh,
            (),
        );
        self.popups.push(popup);
    }

    /// How many popups the server created, as reported by `new_popup`.
    fn created_popups(&self) -> usize {
        self.server.lock().unwrap().popups
    }

    /// Wait until the server has processed every request sent so far. Returns the protocol error
    /// the server answered with, as the error code and the interface it was raised on.
    fn protocol_error(&mut self) -> Option<(u32, String)> {
        match self.queue.roundtrip(&mut self.client) {
            Ok(_) => None,
            Err(DispatchError::Backend(WaylandError::Protocol(error))) => {
                Some((error.code, error.object_interface))
            }
            Err(other) => panic!("unexpected dispatch error: {other:?}"),
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn invalid_popup_parent() -> Option<(u32, String)> {
    Some((
        xdg_wm_base::Error::InvalidPopupParent as u32,
        "xdg_wm_base".to_owned(),
    ))
}

// ----------------------------------------------------------------- tests

/// The sequence of randfall's `xdg_popup_parent_cycle`: two surfaces become popups of each other.
#[test]
fn popups_that_are_each_others_parent_are_a_protocol_error() {
    let mut h = Harness::new();
    let foo = h.surface();
    let bar = h.surface();

    h.popup(foo, bar);
    h.popup(bar, foo);

    assert_eq!(h.protocol_error(), invalid_popup_parent());
    // Only the first request created a popup
    assert_eq!(h.created_popups(), 1);
}

#[test]
fn popup_that_is_its_own_parent_is_a_protocol_error() {
    let mut h = Harness::new();
    let foo = h.surface();

    h.popup(foo, foo);

    assert_eq!(h.protocol_error(), invalid_popup_parent());
    assert_eq!(h.created_popups(), 0);
}

#[test]
fn longer_popup_cycle_is_a_protocol_error() {
    let mut h = Harness::new();
    let a = h.surface();
    let b = h.surface();
    let c = h.surface();

    h.popup(a, b);
    h.popup(b, c);
    h.popup(c, a);

    assert_eq!(h.protocol_error(), invalid_popup_parent());
    assert_eq!(h.created_popups(), 2);
}

/// Control: popups nested below a toplevel are accepted.
#[test]
fn chain_of_popups_below_a_toplevel_is_accepted() {
    let mut h = Harness::new();
    let root = h.surface();
    let first = h.surface();
    let second = h.surface();
    h.toplevel(root);

    h.popup(first, root);
    h.popup(second, first);

    assert_eq!(h.protocol_error(), None);
    assert_eq!(h.created_popups(), 2);
}

/// Control: the parent only has to get its role before the popup is mapped, so a parent without a
/// role when the popup is created is not an error by itself.
#[test]
fn popup_parent_may_get_its_role_after_the_popup_is_created() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();

    h.popup(child, parent);
    h.toplevel(parent);

    assert_eq!(h.protocol_error(), None);
    assert_eq!(h.created_popups(), 1);
}

/// A surface that already is a toplevel gets the role error, not the popup parent error, even
/// when the requested parent is the surface itself.
#[test]
fn popup_request_on_a_toplevel_keeps_the_role_error() {
    let mut h = Harness::new();
    let root = h.surface();
    h.toplevel(root);

    h.popup(root, root);

    assert_eq!(
        h.protocol_error(),
        Some((xdg_wm_base::Error::Role as u32, "xdg_wm_base".to_owned()))
    );
    assert_eq!(h.created_popups(), 0);
}
