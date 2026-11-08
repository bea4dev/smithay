//! Tests for the `wl_subsurface` object of a `wl_surface`.
//!
//! A `wl_surface` must not get a second `wl_subsurface` object while the first one exists, also
//! when the parent of the first one is gone. These tests drive a real `wayland-client` connection
//! against a live [`Display`], so the requests go through the protocol handlers.

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
    protocol::{wl_compositor, wl_registry, wl_subcompositor, wl_subsurface, wl_surface},
};
use wayland_server::{Client, Display, DisplayHandle, backend::ClientData, protocol::wl_surface::WlSurface};

use crate::wayland::compositor::{CompositorClientState, CompositorHandler, CompositorState};

// ---------------------------------------------------------------- server

#[derive(Default)]
struct ServerClient {
    compositor: CompositorClientState,
}

impl ClientData for ServerClient {}

struct Server {
    compositor: CompositorState,
    /// Number of subsurfaces the server was told about with `new_subsurface`.
    subsurfaces: usize,
}

impl CompositorHandler for Server {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ServerClient>().unwrap().compositor
    }

    fn new_subsurface(&mut self, _surface: &WlSurface, _parent: &WlSurface) {
        self.subsurfaces += 1;
    }

    fn commit(&mut self, _surface: &WlSurface) {}
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
delegate_noop!(ClientApp: wl_subcompositor::WlSubcompositor);
delegate_noop!(ClientApp: ignore wl_surface::WlSurface);
delegate_noop!(ClientApp: wl_subsurface::WlSubsurface);

// --------------------------------------------------------------- harness

struct Harness {
    server: Arc<Mutex<Server>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    queue: EventQueue<ClientApp>,
    qh: QueueHandle<ClientApp>,
    client: ClientApp,
    compositor: wl_compositor::WlCompositor,
    subcompositor: wl_subcompositor::WlSubcompositor,
    /// Client side objects, which must stay alive until a test destroys them.
    surfaces: Vec<wl_surface::WlSurface>,
    subsurfaces: Vec<wl_subsurface::WlSubsurface>,
}

impl Harness {
    fn new() -> Self {
        let mut display = Display::<Server>::new().unwrap();
        let mut dh: DisplayHandle = display.handle();
        let server = Arc::new(Mutex::new(Server {
            compositor: CompositorState::new::<Server>(&dh),
            subsurfaces: 0,
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
        let subcompositor = globals.bind(&qh, 1..=1, ()).unwrap();

        Harness {
            server,
            stop,
            thread: Some(thread),
            queue,
            qh,
            client: ClientApp,
            compositor,
            subcompositor,
            surfaces: Vec::new(),
            subsurfaces: Vec::new(),
        }
    }

    /// Create a `wl_surface`; returns its index.
    fn surface(&mut self) -> usize {
        let surface = self.compositor.create_surface(&self.qh, ());
        self.surfaces.push(surface);
        self.surfaces.len() - 1
    }

    /// Ask for `child` to become a subsurface of `parent`; returns the index of the new
    /// `wl_subsurface` object.
    fn subsurface(&mut self, child: usize, parent: usize) -> usize {
        let subsurface =
            self.subcompositor
                .get_subsurface(&self.surfaces[child], &self.surfaces[parent], &self.qh, ());
        self.subsurfaces.push(subsurface);
        self.subsurfaces.len() - 1
    }

    /// How many subsurfaces the server was told about, as reported by `new_subsurface`.
    fn created_subsurfaces(&self) -> usize {
        self.server.lock().unwrap().subsurfaces
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

fn bad_surface() -> Option<(u32, String)> {
    Some((
        wl_subcompositor::Error::BadSurface as u32,
        "wl_subcompositor".to_owned(),
    ))
}

// ----------------------------------------------------------------- tests

/// The sequence of randfall's `subsurface_double_after_parent_destroyed`: the parent of the first
/// `wl_subsurface` object is destroyed, but the object still exists.
#[test]
fn second_subsurface_object_after_the_parent_was_destroyed_is_a_protocol_error() {
    let mut h = Harness::new();
    let first_parent = h.surface();
    let second_parent = h.surface();
    let child = h.surface();

    h.subsurface(child, first_parent);
    h.surfaces[first_parent].destroy();
    h.subsurface(child, second_parent);

    assert_eq!(h.protocol_error(), bad_surface());
    // Only the first request created a subsurface
    assert_eq!(h.created_subsurfaces(), 1);
}

/// Control: this was already rejected, by the parent that the first object still has.
#[test]
fn second_subsurface_object_while_the_first_one_has_a_parent_is_a_protocol_error() {
    let mut h = Harness::new();
    let parent = h.surface();
    let other_parent = h.surface();
    let child = h.surface();

    h.subsurface(child, parent);
    h.subsurface(child, other_parent);

    assert_eq!(h.protocol_error(), bad_surface());
    assert_eq!(h.created_subsurfaces(), 1);
}

/// Control: once the `wl_subsurface` object is destroyed the surface can get a new one, also
/// when the parent of the first one had been destroyed before.
#[test]
fn subsurface_object_can_be_created_again_after_the_first_one_was_destroyed() {
    let mut h = Harness::new();
    let first_parent = h.surface();
    let second_parent = h.surface();
    let child = h.surface();

    let first = h.subsurface(child, first_parent);
    h.surfaces[first_parent].destroy();
    h.subsurfaces[first].destroy();
    h.subsurface(child, second_parent);

    assert_eq!(h.protocol_error(), None);
    assert_eq!(h.created_subsurfaces(), 2);
}

/// Control: the same without the parent being destroyed.
#[test]
fn subsurface_object_can_be_created_again_after_the_first_one_was_destroyed_with_its_parent_alive() {
    let mut h = Harness::new();
    let first_parent = h.surface();
    let second_parent = h.surface();
    let child = h.surface();

    let first = h.subsurface(child, first_parent);
    h.subsurfaces[first].destroy();
    h.subsurface(child, second_parent);

    assert_eq!(h.protocol_error(), None);
    assert_eq!(h.created_subsurfaces(), 2);
}
