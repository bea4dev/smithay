//! Regression tests for the commit semantics of synchronized subsurfaces.
//!
//! These tests drive a real `wayland-client` connection against a live
//! [`Display`], so every assertion goes through the protocol handlers and the
//! transaction machinery, not through hand-built state.
//!
//! The rule being checked is the one from `wl_subsurface`: the state of a
//! synchronized subsurface is applied when its parent commits, but only the
//! state the subsurface itself has committed. The position set with
//! `wl_subsurface.set_position` belongs to the parent and takes effect on the
//! parent's next commit (for a desynchronized subsurface Smithay applies it with
//! the subsurface's own commit, which these tests do not pin down).

use std::{
    os::{fd::AsFd, unix::net::UnixStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use wayland_client::{
    Connection, Dispatch, EventQueue, QueueHandle, delegate_noop,
    globals::{GlobalListContents, registry_queue_init},
    protocol::{
        wl_buffer, wl_compositor, wl_registry, wl_shm, wl_shm_pool, wl_subcompositor, wl_subsurface,
        wl_surface,
    },
};
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};
use wayland_server::{
    Client, Display, DisplayHandle,
    backend::ClientData,
    protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface},
};

use crate::{
    utils::{Logical, Point, Size},
    wayland::{
        buffer::BufferHandler,
        compositor::{
            BufferAssignment, CompositorClientState, CompositorHandler, CompositorState,
            SubsurfaceCachedState, SurfaceAttributes, with_states,
        },
        shm::{ShmHandler, ShmState},
        viewporter::{ViewportCachedState, ViewporterState},
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
    _viewporter: ViewporterState,
    shm: ShmState,
    /// Server side surfaces in creation order.
    surfaces: Vec<WlSurface>,
}

impl CompositorHandler for Server {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ServerClient>().unwrap().compositor
    }

    fn new_surface(&mut self, surface: &WlSurface) {
        self.surfaces.push(surface.clone());
    }

    fn commit(&mut self, _surface: &WlSurface) {}
}

impl BufferHandler for Server {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}

impl ShmHandler for Server {
    fn shm_state(&self) -> &ShmState {
        &self.shm
    }
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
delegate_noop!(ClientApp: wp_viewporter::WpViewporter);
delegate_noop!(ClientApp: wp_viewport::WpViewport);
delegate_noop!(ClientApp: ignore wl_shm::WlShm);
delegate_noop!(ClientApp: wl_shm_pool::WlShmPool);
delegate_noop!(ClientApp: ignore wl_buffer::WlBuffer);

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
    viewporter: wp_viewporter::WpViewporter,
    shm: wl_shm::WlShm,
    /// Client side surfaces, same order as `Server::surfaces`.
    surfaces: Vec<wl_surface::WlSurface>,
    /// Client side objects that must stay alive for the whole test.
    subsurfaces: Vec<wl_subsurface::WlSubsurface>,
    viewports: Vec<Option<wp_viewport::WpViewport>>,
}

impl Harness {
    fn new() -> Self {
        let mut display = Display::<Server>::new().unwrap();
        let mut dh: DisplayHandle = display.handle();
        let server = Arc::new(Mutex::new(Server {
            compositor: CompositorState::new::<Server>(&dh),
            _viewporter: ViewporterState::new::<Server>(&dh),
            shm: ShmState::new::<Server>(&dh, []),
            surfaces: Vec::new(),
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
                        display.dispatch_clients(&mut *server).unwrap();
                        display.flush_clients().unwrap();
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
        let viewporter = globals.bind(&qh, 1..=1, ()).unwrap();
        let shm = globals.bind(&qh, 1..=1, ()).unwrap();

        Harness {
            server,
            stop,
            thread: Some(thread),
            queue,
            qh,
            client: ClientApp,
            compositor,
            subcompositor,
            viewporter,
            shm,
            surfaces: Vec::new(),
            subsurfaces: Vec::new(),
            viewports: Vec::new(),
        }
    }

    /// Wait until the server has processed every request sent so far.
    fn sync(&mut self) {
        self.queue.roundtrip(&mut self.client).unwrap();
    }

    /// Create a surface; returns its index.
    fn surface(&mut self) -> usize {
        let surface = self.compositor.create_surface(&self.qh, ());
        self.surfaces.push(surface);
        self.viewports.push(None);
        self.sync();
        self.surfaces.len() - 1
    }

    /// Make `child` a (synchronized) subsurface of `parent`.
    fn subsurface(&mut self, child: usize, parent: usize) -> usize {
        let sub =
            self.subcompositor
                .get_subsurface(&self.surfaces[child], &self.surfaces[parent], &self.qh, ());
        self.subsurfaces.push(sub);
        self.sync();
        self.subsurfaces.len() - 1
    }

    fn set_destination(&mut self, surface: usize, width: i32, height: i32) {
        if self.viewports[surface].is_none() {
            let viewport = self
                .viewporter
                .get_viewport(&self.surfaces[surface], &self.qh, ());
            self.viewports[surface] = Some(viewport);
        }
        self.viewports[surface]
            .as_ref()
            .unwrap()
            .set_destination(width, height);
        self.sync();
    }

    /// Attach a freshly allocated `width` x `height` shm buffer to the surface (not committed).
    fn attach_buffer(&mut self, surface: usize, width: i32, height: i32) {
        use rustix::fs::{MemfdFlags, ftruncate, memfd_create};

        let stride = width * 4;
        let len = (stride * height) as u64;
        let fd = memfd_create("sync-subsurface-test", MemfdFlags::CLOEXEC).unwrap();
        ftruncate(&fd, len).unwrap();
        let pool = self.shm.create_pool(fd.as_fd(), len as i32, &self.qh, ());
        let buffer = pool.create_buffer(0, width, height, stride, wl_shm::Format::Argb8888, &self.qh, ());
        pool.destroy();
        self.surfaces[surface].attach(Some(&buffer), 0, 0);
        self.sync();
    }

    fn commit(&mut self, surface: usize) {
        self.surfaces[surface].commit();
        self.sync();
    }

    /// Whether a buffer is currently applied to the surface.
    fn has_applied_buffer(&self, surface: usize) -> bool {
        with_states(&self.server_surface(surface), |states| {
            matches!(
                states.cached_state.get::<SurfaceAttributes>().current().buffer,
                Some(BufferAssignment::NewBuffer(_))
            )
        })
    }

    /// Currently applied buffer scale of a surface.
    fn applied_buffer_scale(&self, surface: usize) -> i32 {
        with_states(&self.server_surface(surface), |states| {
            states
                .cached_state
                .get::<SurfaceAttributes>()
                .current()
                .buffer_scale
        })
    }

    fn server_surface(&self, index: usize) -> WlSurface {
        self.server.lock().unwrap().surfaces[index].clone()
    }

    /// Currently applied destination size of a surface's viewport.
    fn applied_destination(&self, surface: usize) -> Option<Size<i32, Logical>> {
        with_states(&self.server_surface(surface), |states| {
            states.cached_state.get::<ViewportCachedState>().current().dst
        })
    }

    /// Currently applied subsurface location.
    fn applied_location(&self, surface: usize) -> Point<i32, Logical> {
        with_states(&self.server_surface(surface), |states| {
            states
                .cached_state
                .get::<SubsurfaceCachedState>()
                .current()
                .location
        })
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

fn size(w: i32, h: i32) -> Option<Size<i32, Logical>> {
    Some(Size::from((w, h)))
}

// ----------------------------------------------------------------- tests

/// Positive control: the state a synchronized subsurface committed is held
/// back until the parent commits, then applied.
#[test]
fn committed_child_state_is_applied_by_parent_commit() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.set_destination(child, 64, 64);
    h.commit(child);
    assert_eq!(
        h.applied_destination(child),
        None,
        "applied before the parent committed"
    );

    h.commit(parent);
    assert_eq!(h.applied_destination(child), size(64, 64));
}

/// Negative: state the child never committed must not be applied.
#[test]
fn parent_commit_does_not_apply_never_committed_child_state() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.set_destination(child, 100, 100);
    h.commit(parent);

    assert_eq!(h.applied_destination(child), None);
}

/// Negative: a pending change after the child's last commit must not replace
/// the already applied state.
#[test]
fn parent_commit_keeps_applied_child_state_over_pending_change() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.set_destination(child, 64, 64);
    h.commit(child);
    h.commit(parent);
    assert_eq!(h.applied_destination(child), size(64, 64));

    h.set_destination(child, 100, 100);
    h.commit(parent);

    assert_eq!(h.applied_destination(child), size(64, 64));
}

/// Negative: a pending change must not replace a committed snapshot that is
/// still waiting for the parent.
#[test]
fn parent_commit_applies_committed_snapshot_not_newer_pending_state() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.set_destination(child, 120, 120);
    h.commit(child);
    h.set_destination(child, 160, 160);
    h.commit(parent);

    assert_eq!(h.applied_destination(child), size(120, 120));
}

/// The held back change is not lost: it is applied once the child commits it
/// and the parent commits again.
#[test]
fn pending_child_state_is_applied_after_child_and_parent_commit() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.set_destination(child, 120, 120);
    h.commit(child);
    h.set_destination(child, 160, 160);
    h.commit(parent);
    assert_eq!(h.applied_destination(child), size(120, 120));

    h.commit(child);
    assert_eq!(
        h.applied_destination(child),
        size(120, 120),
        "applied before the parent committed"
    );
    h.commit(parent);
    assert_eq!(h.applied_destination(child), size(160, 160));
}

/// Preserved behaviour: the position belongs to the parent, so for a synchronized child it takes
/// effect on the parent's commit even though the child never committed.
#[test]
fn parent_commit_applies_subsurface_position_without_child_commit() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    let sub = h.subsurface(child, parent);

    h.subsurfaces[sub].set_position(10, 20);
    h.sync();
    assert_eq!(
        h.applied_location(child),
        Point::from((0, 0)),
        "applied before the parent committed"
    );

    h.commit(parent);
    assert_eq!(h.applied_location(child), Point::from((10, 20)));
}

/// Both rules at once: the parent commit applies the parent-owned position
/// and leaves the child-owned pending state alone.
#[test]
fn parent_commit_applies_position_but_not_pending_child_state() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    let sub = h.subsurface(child, parent);

    h.set_destination(child, 64, 64);
    h.commit(child);
    h.set_destination(child, 100, 100);
    h.subsurfaces[sub].set_position(7, 9);
    h.sync();
    h.commit(parent);

    assert_eq!(h.applied_location(child), Point::from((7, 9)));
    assert_eq!(h.applied_destination(child), size(64, 64));
}

/// A desynchronized subsurface is not held back by its parent at all.
#[test]
fn desync_child_state_is_applied_by_its_own_commit() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    let sub = h.subsurface(child, parent);

    h.subsurfaces[sub].set_desync();
    h.sync();
    h.set_destination(child, 64, 64);
    h.commit(child);

    assert_eq!(h.applied_destination(child), size(64, 64));
}

/// Nested synchronized subsurfaces: a commit of the root must not apply the
/// uncommitted state of a grandchild, but still applies its position.
#[test]
fn nested_sync_tree_applies_committed_state_and_positions_only() {
    let mut h = Harness::new();
    let root = h.surface();
    let middle = h.surface();
    let leaf = h.surface();
    h.subsurface(middle, root);
    let leaf_sub = h.subsurface(leaf, middle);

    h.set_destination(leaf, 32, 32);
    h.commit(leaf);
    h.set_destination(leaf, 48, 48);
    h.subsurfaces[leaf_sub].set_position(3, 4);
    h.sync();
    h.commit(middle);
    h.commit(root);

    assert_eq!(h.applied_destination(leaf), size(32, 32));
    assert_eq!(h.applied_location(leaf), Point::from((3, 4)));
}

/// Negative: a buffer the child attached but never committed must not be
/// applied by the parent's commit.
#[test]
fn parent_commit_does_not_apply_uncommitted_child_buffer() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.attach_buffer(child, 8, 8);
    h.commit(parent);

    assert!(!h.has_applied_buffer(child));
}

/// Positive control for the buffer: once the child commits it, the parent's
/// commit applies it, and not before.
#[test]
fn parent_commit_applies_committed_child_buffer() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.attach_buffer(child, 8, 8);
    h.commit(child);
    assert!(
        !h.has_applied_buffer(child),
        "applied before the parent committed"
    );

    h.commit(parent);
    assert!(h.has_applied_buffer(child));
}

/// Negative: `wl_surface.set_buffer_scale` is double-buffered child state too.
#[test]
fn parent_commit_does_not_apply_uncommitted_child_buffer_scale() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.surfaces[child].set_buffer_scale(2);
    h.sync();
    h.commit(parent);

    assert_eq!(h.applied_buffer_scale(child), 1);

    h.commit(child);
    h.commit(parent);
    assert_eq!(h.applied_buffer_scale(child), 2);
}

/// The child commits twice before the parent does: the later committed
/// snapshot wins, and an uncommitted third value is still held back.
#[test]
fn latest_committed_child_snapshot_wins_over_earlier_one() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    h.set_destination(child, 32, 32);
    h.commit(child);
    h.set_destination(child, 48, 48);
    h.commit(child);
    h.set_destination(child, 96, 96);
    h.commit(parent);

    assert_eq!(h.applied_destination(child), size(48, 48));
}

/// The rule holds across repeated child and parent commit cycles.
#[test]
fn repeated_commit_cycles_apply_only_committed_state() {
    let mut h = Harness::new();
    let parent = h.surface();
    let child = h.surface();
    h.subsurface(child, parent);

    for step in 1..=3 {
        let committed = 10 * step;
        h.set_destination(child, committed, committed);
        h.commit(child);
        h.set_destination(child, committed + 5, committed + 5);
        h.commit(parent);
        assert_eq!(
            h.applied_destination(child),
            size(committed, committed),
            "cycle {step}"
        );
    }
}

/// Siblings are independent: only the child that committed has state applied.
#[test]
fn only_the_sibling_that_committed_gets_its_state_applied() {
    let mut h = Harness::new();
    let parent = h.surface();
    let first = h.surface();
    let second = h.surface();
    h.subsurface(first, parent);
    h.subsurface(second, parent);

    h.set_destination(first, 20, 20);
    h.commit(first);
    h.set_destination(second, 30, 30);
    h.commit(parent);

    assert_eq!(h.applied_destination(first), size(20, 20));
    assert_eq!(h.applied_destination(second), None);
}

/// A desynchronized surface in the middle of a tree: committing the root
/// reaches its synchronized child, which keeps its uncommitted state back
/// while its position is applied.
#[test]
fn desync_middle_surface_does_not_leak_uncommitted_leaf_state() {
    let mut h = Harness::new();
    let root = h.surface();
    let middle = h.surface();
    let leaf = h.surface();
    let middle_sub = h.subsurface(middle, root);
    let leaf_sub = h.subsurface(leaf, middle);

    h.subsurfaces[middle_sub].set_desync();
    h.sync();
    h.set_destination(leaf, 32, 32);
    h.commit(leaf);
    h.set_destination(leaf, 48, 48);
    h.subsurfaces[leaf_sub].set_position(5, 6);
    h.sync();
    h.commit(middle);

    assert_eq!(h.applied_destination(leaf), size(32, 32));
    assert_eq!(h.applied_location(leaf), Point::from((5, 6)));
}
