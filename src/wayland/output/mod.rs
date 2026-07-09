//! WlOutput advertising capabilities
//!
//! This module provides additional implementations for [`crate::output::Output`]
//! to handle advertising of your compositor's output and their capabilities to your client,
//! as well as mapping your clients output request to your physical outputs.
//!
//! # How to use it
//!
//! After you have instantiated an [`Output`] you need to use [`Output::create_global`]
//! to advertise a new output global to clients.
//! The resulting `GlobalId` can later be destroyed again to stop advertising it
//! without destroying it's state. E.g. in case the matching physical output got disabled at runtime.
//!
//! If you change the properties of your output (if the current resolution mode changes for example),
//! it'll automatically forward any changes to the clients.
//!
//! Additional protocols may piggy-back on this type.
//! E.g. to also advertise an xdg-output for every wl-output you can use
//! [`OutputManagerState::new_with_xdg_output`].
//!
//! ```
//! # extern crate wayland_server;
//! # extern crate smithay;
//! use smithay::output::{Output, PhysicalProperties, Scale, Mode, Subpixel};
//! use smithay::utils::Transform;
//! use smithay::wayland::output::OutputHandler;
//! # use smithay::wayland::compositor::{CompositorHandler, CompositorState, CompositorClientState};
//! # use smithay::reexports::wayland_server::{Client, protocol::wl_surface::WlSurface};
//!
//! # struct State;
//! # let mut display = wayland_server::Display::<State>::new().unwrap();
//! # let display_handle = display.handle();
//! // Create the Output with given name and physical properties.
//! let output = Output::new(
//!     "output-0".into(), // the name of this output,
//!     PhysicalProperties {
//!         size: (200, 150).into(),        // dimensions (width, height) in mm
//!         subpixel: Subpixel::HorizontalRgb,  // subpixel information
//!         make: "Screens Inc".into(),     // make of the monitor
//!         model: "Monitor Ultra".into(),  // model of the monitor
//!         serial_number: "KL11C28J04XJ".into(), // serial number of the monitor
//!     },
//! );
//! // create a global, if you want to advertise it to clients
//! let _global = output.create_global::<State>(
//!     &display_handle,      // the display
//! ); // you can drop the global, if you never intend to destroy it.
//! // Now you can configure it
//! output.change_current_state(
//!     Some(Mode { size: (1920, 1080).into(), refresh: 60000 }), // the resolution mode,
//!     Some(Transform::Normal), // global screen transformation
//!     Some(Scale::Integer(1)), // global screen scaling factor
//!     Some((0,0).into()) // output position
//! );
//! // set the preferred mode
//! output.set_preferred(Mode { size: (1920, 1080).into(), refresh: 60000 });
//! // add other supported modes
//! output.add_mode(Mode { size: (800, 600).into(), refresh: 60000 });
//! output.add_mode(Mode { size: (1024, 768).into(), refresh: 60000 });
//!
//! impl OutputHandler for State {}
//! # impl CompositorHandler for State {
//! #     fn compositor_state(&mut self) -> &mut CompositorState { unimplemented!() }
//! #     fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState { unimplemented!() }
//! #     fn commit(&mut self, surface: &WlSurface) {}
//! # }
//!
//! smithay::delegate_dispatch2!(State);
//! ```

mod handlers;
pub(crate) mod xdg;

use std::mem;
use std::sync::{Arc, atomic::Ordering};

use crate::{
    output::{Inner, Mode, Output, Scale, Subpixel, WeakOutput},
    utils::iter::new_locked_obj_iter,
};

use atomic_float::AtomicF64;
use tracing::info;
use wayland_protocols::xdg::xdg_output::zv1::server::zxdg_output_manager_v1::ZxdgOutputManagerV1;
use wayland_server::{
    Client, DisplayHandle, GlobalDispatch, Resource,
    backend::{ClientId, GlobalId},
    protocol::{
        wl_output::{Mode as WMode, Subpixel as WlSubpixel, Transform, WlOutput},
        wl_surface,
    },
};

use crate::{
    utils::{Logical, Point},
    wayland::GlobalData,
};

pub use self::handlers::XdgOutputUserData;

/// State of Smithay output manager
#[derive(Debug, Default)]
pub struct OutputManagerState {
    xdg_output_manager: Option<GlobalId>,
}

/// Callback used to override the `wl_output` state advertised to a client.
///
/// The callback is evaluated for each client that binds a `wl_output` and each time state is
/// re-sent.
///
/// `None`: preserves the output's physical state.
/// `Some(ClientOutputOverride)`: applies the override for that client only.
///
/// Most compositors should not need this. It is intended for compatibility layers where a single
/// Wayland client represents windows on multiple outputs and mirrors `wl_output` data into RandR
/// at face value.
///
/// Required by:
///
/// Xwayland
/// Xwayland-satellite
pub type ModeRefreshOverride = dyn Fn(&Client) -> Option<ClientOutputOverride> + Send + Sync + 'static;

/// Per-client overrides of the state a `wl_output` global advertises.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClientOutputOverride {
    /// Synthesized refresh in mHz. `None` reports the physical refresh.
    pub refresh_mhz: Option<i32>,
    /// Advertise every mode at the output's *logical* size (physical size divided by the
    /// output scale) and report `wl_output.scale` as 1.
    ///
    /// This makes the client's coordinate world identical to the compositor's logical layout.
    /// X11 bridges mirror `wl_output.mode` into RandR at face value while taking positions from
    /// `xdg_output` (logical), so on fractionally scaled outputs they would otherwise construct
    /// an overlapping, skewed X11 screen — visible as offset pointer input in X11 clients.
    pub normalize_to_logical: bool,
}


/// Internal data of a wl_output global
pub struct WlOutputData {
    output: Output,
    mode_refresh_override: Option<Arc<ModeRefreshOverride>>,
}

impl std::fmt::Debug for WlOutputData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WlOutputData")
            .field("output", &self.output)
            .field("mode_refresh_override", &self.mode_refresh_override.is_some())
            .finish()
    }
}

/// Events initiated by the clients interacting with outputs
pub trait OutputHandler {
    /// A client bound a new `wl_output` instance.
    fn output_bound(&mut self, _output: Output, _wl_output: WlOutput) {}
}

impl OutputManagerState {
    /// Create new output manager
    pub fn new() -> Self {
        Self {
            xdg_output_manager: None,
        }
    }

    /// Create new output manager with xdg output support
    pub fn new_with_xdg_output<D>(display: &DisplayHandle) -> Self
    where
        D: GlobalDispatch<WlOutput, WlOutputData>,
        D: GlobalDispatch<ZxdgOutputManagerV1, GlobalData>,
        D: 'static,
    {
        let xdg_output_manager = display.create_global::<D, ZxdgOutputManagerV1, _>(3, GlobalData);

        Self {
            xdg_output_manager: Some(xdg_output_manager),
        }
    }

    /// Get global id of xdg output manager
    pub fn xdg_output_manager_global(&self) -> Option<GlobalId> {
        self.xdg_output_manager.clone()
    }
}

/// User data for WlOutput
pub struct OutputUserData {
    pub(crate) output: WeakOutput,
    last_client_scale: AtomicF64,
    client_scale: Arc<AtomicF64>,
    mode_refresh_override: Option<Arc<ModeRefreshOverride>>,
}

impl std::fmt::Debug for OutputUserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputUserData")
            .field("output", &self.output)
            .field(
                "last_client_scale",
                &self.last_client_scale.load(Ordering::Acquire),
            )
            .field("client_scale", &self.client_scale.load(Ordering::Acquire))
            .field("mode_refresh_override", &self.mode_refresh_override.is_some())
            .finish()
    }
}

impl Clone for OutputUserData {
    fn clone(&self) -> Self {
        OutputUserData {
            output: self.output.clone(),
            last_client_scale: AtomicF64::new(self.last_client_scale.load(Ordering::Acquire)),
            client_scale: self.client_scale.clone(),
            mode_refresh_override: self.mode_refresh_override.clone(),
        }
    }
}

impl OutputUserData {
    pub(crate) fn client_override_for(&self, output: &WlOutput) -> Option<ClientOutputOverride> {
        self.mode_refresh_override
            .as_ref()
            .and_then(|override_fn| output.client().and_then(|client| override_fn(&client)))
    }

    /// The `(width, height, refresh)` to advertise for `mode` on this instance, honoring the
    /// per-client override. `scale` is the output's current scale, used when normalizing the
    /// mode to logical size.
    pub(crate) fn mode_event_for(&self, output: &WlOutput, mode: Mode, scale: Scale) -> (i32, i32, i32) {
        match self.client_override_for(output) {
            Some(client_override) => {
                let (w, h) = if client_override.normalize_to_logical {
                    let factor = scale.fractional_scale();
                    (
                        ((mode.size.w as f64 / factor).round() as i32).max(1),
                        ((mode.size.h as f64 / factor).round() as i32).max(1),
                    )
                } else {
                    (mode.size.w, mode.size.h)
                };
                (
                    w,
                    h,
                    client_override.refresh_mhz
                        .unwrap_or(
                            mode.refresh
                        ),
                )
            }
            None => (mode.size.w, mode.size.h, mode.refresh),
        }
    }

    /// The `wl_output.scale` value to advertise on this instance.
    pub(crate) fn scale_event_for(&self, output: &WlOutput, scale: i32) -> i32 {
        if self
            .client_override_for(output)
            .is_some_and(
                |client_override| client_override.normalize_to_logical
            )
        {
            1
        } else {
            scale
        }
    }
}

impl Inner {
    fn send_geometry_to(&self, output: &WlOutput) {
        output.geometry(
            self.location.x,
            self.location.y,
            self.physical.size.w,
            self.physical.size.h,
            self.physical.subpixel.into(),
            self.physical.make.clone(),
            self.physical.model.clone(),
            self.transform.into(),
        );
    }
}

impl From<Subpixel> for WlSubpixel {
    #[inline]
    fn from(s: Subpixel) -> Self {
        match s {
            Subpixel::HorizontalBgr => WlSubpixel::HorizontalBgr,
            Subpixel::HorizontalRgb => WlSubpixel::HorizontalRgb,
            Subpixel::None => WlSubpixel::None,
            Subpixel::Unknown => WlSubpixel::Unknown,
            Subpixel::VerticalBgr => WlSubpixel::VerticalBgr,
            Subpixel::VerticalRgb => WlSubpixel::VerticalRgb,
        }
    }
}

impl Output {
    /// Create a new output global.
    ///
    /// The global is directly registered into the event loop, and this function
    /// returns the global handle in case you wish to remove this global in the future.
    ///
    /// Calling this function multiple times without destroying the global in between,
    /// will result in multiple globals, meaning the output will be advertised to clients
    /// multiple times.
    pub fn create_global<D>(&self, display: &DisplayHandle) -> GlobalId
    where
        D: GlobalDispatch<WlOutput, WlOutputData>,
        D: 'static,
    {
        self.create_global_internal::<D>(display, None)
    }

    /// Create a new output global with a per-client mode refresh override.
    ///
    /// This is a compatibility escape hatch for clients that do not map cleanly to a single
    /// physical output. For example, Xwayland is one Wayland client but may expose X11 windows on
    /// several outputs; some X11 clients then pace rendering from RandR/Present using a single
    /// representative CRTC refresh. A compositor can use this hook to synthesize the refresh seen
    /// by that compatibility client while all regular clients continue to see the physical output
    /// mode unchanged.
    ///
    /// The callback is evaluated whenever Smithay sends `wl_output` state for this global.
    ///
    /// `None`: normal clients
    /// `Some(ClientOutputOverride)`: synthesizes refresh and/or normalize mode sizes and scale
    /// to the logical coordinate space for that client (see [`ClientOutputOverride`]).
    ///
    /// If the override value changes without a physical output state change, call
    /// [`Output::change_current_state`] with the current mode to re-send the mode event.
    pub fn create_global_with_mode_refresh_override<D, F>(
        &self,
        display: &DisplayHandle,
        mode_refresh_override: F,
    ) -> GlobalId
    where
        D: GlobalDispatch<WlOutput, WlOutputData>,
        D: 'static,
        F: Fn(&Client) -> Option<
            ClientOutputOverride
        > + Send + Sync + 'static,
    {
        self.create_global_internal::<D>(
            display,
            Some(
                Arc::new(
                    mode_refresh_override,
                )
            ),
        )
    }

    fn create_global_internal<D>(
        &self,
        display: &DisplayHandle,
        mode_refresh_override: Option<Arc<ModeRefreshOverride>>,
    ) -> GlobalId
    where
        D: GlobalDispatch<WlOutput, WlOutputData>,
        D: 'static,
    {
        info!(output = self.name(), "Creating new wl_output");
        self.inner.0.lock().unwrap().handle = Some(display.backend_handle().downgrade());
        display.create_global::<D, WlOutput, _>(
            4,
            WlOutputData {
                output: self.clone(),
                mode_refresh_override,
            },
        )
    }

    /// Attempt to retrieve a [`Output`] from an existing resource
    pub fn from_resource(output: &WlOutput) -> Option<Output> {
        output.data::<OutputUserData>().and_then(|ud| ud.output.upgrade())
    }

    pub(crate) fn wl_change_current_state(
        &self,
        new_mode: Option<Mode>,
        new_transform: Option<Transform>,
        new_scale: Option<Scale>,
        new_location: Option<Point<i32, Logical>>,
    ) {
        let inner = self.inner.0.lock().unwrap();
        // XdgOutput has to be updated before WlOutput
        // Because WlOutput::done() has to always be called last
        if let Some(xdg_output) = inner.xdg_output.as_ref() {
            xdg_output.change_current_state(new_mode, new_scale, new_location, new_transform);
        }

        let mut flags = WMode::Current;
        if inner.preferred_mode == new_mode {
            flags |= WMode::Preferred;
        }

        for output in &inner.instances {
            let Ok(output) = output.upgrade() else {
                continue;
            };

            let data = output.data::<OutputUserData>().unwrap();
            let client_scale = data.client_scale.load(Ordering::Acquire);
            let scale_changed = client_scale != data.last_client_scale.swap(client_scale, Ordering::AcqRel);

            if let Some(mode) = new_mode {
                output.mode(
                    flags,
                    mode.size.w,
                    mode.size.h,
                    data.mode_refresh_for(&output, mode.refresh),
                );
            }
            if new_transform.is_some() || new_location.is_some() {
                inner.send_geometry_to(&output);
            }
            if (new_scale.is_some() || scale_changed) && output.version() >= 2 {
                let scale = (inner.scale.integer_scale() as f64 / client_scale).max(1.).ceil() as i32;
                output.scale(scale);
            }
            if output.version() >= 2 {
                output.done();
            }
        }
    }

    /// Check is given [`wl_output`](WlOutput) instance is managed by this [`Output`].
    pub fn owns(&self, output: &WlOutput) -> bool {
        self.inner
            .0
            .lock()
            .unwrap()
            .instances
            .iter()
            .any(|o| o.id() == output.id())
    }

    /// This function returns all managed [WlOutput] matching the provided [Client]
    pub fn client_outputs<'a>(&'a self, client: &Client) -> impl Iterator<Item = WlOutput> + 'a {
        self.client_outputs_internal(client.id())
    }

    fn client_outputs_internal(&self, client: ClientId) -> impl Iterator<Item = WlOutput> + '_ {
        let guard = self.inner.0.lock().unwrap();

        new_locked_obj_iter(guard, client, |inner| inner.instances.iter())
    }

    /// Sends `wl_surface.enter` for the provided surface
    /// with the matching client output
    #[profiling::function]
    pub fn enter(&self, surface: &wl_surface::WlSurface) {
        let mut inner = self.inner.0.lock().unwrap();
        if inner.surfaces.insert(surface.downgrade()) {
            let client = inner
                .handle
                .as_ref()
                .and_then(|handle| handle.upgrade())
                .and_then(|handle| handle.get_client(surface.id()).ok());
            drop(inner);

            if let Some(client) = client {
                for output in self.client_outputs_internal(client) {
                    surface.enter(&output);
                }
            }
        }
    }

    /// Sends `wl_surface.leave` for the provided surface
    /// with the matching client output
    #[profiling::function]
    pub fn leave(&self, surface: &wl_surface::WlSurface) {
        let mut inner = self.inner.0.lock().unwrap();
        if inner.surfaces.remove(&surface.downgrade()) {
            let client = inner
                .handle
                .as_ref()
                .and_then(|handle| handle.upgrade())
                .and_then(|handle| handle.get_client(surface.id()).ok());
            drop(inner);

            if let Some(client) = client {
                for output in self.client_outputs_internal(client) {
                    surface.leave(&output);
                }
            }
        }
    }

    /// Sends `wl_surface.leave` for all surfaces on output
    pub fn leave_all(&self) {
        let mut inner = self.inner.0.lock().unwrap();
        #[allow(clippy::mutable_key_type)]
        let surfaces = mem::take(&mut inner.surfaces);
        let Some(handle) = inner.handle.as_ref().and_then(|handle| handle.upgrade()) else {
            return;
        };
        drop(inner);

        for surface in surfaces {
            let Ok(surface) = surface.upgrade() else {
                continue;
            };

            if let Ok(client) = handle.get_client(surface.id()) {
                for output in self.client_outputs_internal(client) {
                    surface.leave(&output);
                }
            }
        }
    }

    pub(crate) fn cleanup_surfaces(&self) {
        let mut inner = self.inner.0.lock().unwrap();
        inner.surfaces.retain(|s| s.is_alive());
    }
}
