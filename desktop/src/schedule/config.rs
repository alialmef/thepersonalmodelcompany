//! Placeholder for per-source override config — pause flags, exclusion
//! lists, custom cadence. Populated as the menu-bar control UI lands.

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SourceOverride {
    pub paused: bool,
    pub excluded_people: Vec<String>,
    pub excluded_threads: Vec<String>,
}
