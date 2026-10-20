//! hexe geo (L0): `Detail` + attributes + groups + Houdini `.geo` readers/writer. Hostile input is the default (AGENTS.md §4).
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    clippy::todo,
    clippy::unreachable
)]

mod bgeo_legacy;
#[cfg(test)]
mod bgeo_legacy_tests;
mod bjson;
#[cfg(test)]
mod bjson_golden_tests;
#[cfg(test)]
mod bjson_tests;
#[cfg(test)]
mod bjson_write_tests;
mod blosc;
#[cfg(test)]
mod blosc_tests;
mod detail;
pub mod endorphin;
pub mod endorphin_behaviour;
pub mod endorphin_geo;
#[cfg(test)]
mod endorphin_tests;
mod error;
#[cfg(test)]
mod fidelity_tests;
mod gpd;
mod io;
#[cfg(test)]
mod io_tests;
mod json_read;
mod json_write;
mod nurbs;
#[cfg(test)]
mod nurbs_tests;
mod obj;
mod ply;
#[cfg(test)]
mod ship_tests;
#[cfg(test)]
mod tests;
mod types;

pub use bgeo_legacy::read_geo_bgeo_legacy;
pub use bjson::{bjson_to_value, read_geo_bjson, value_to_bjson, write_geo_bjson};
pub use blosc::{read_geo_sc, sc_decompress};
pub use error::GeoError;
pub use gpd::read_geo_gpd;
pub use io::{is_endorphin_path, read_any, read_any_at, read_endorphin_file, write_any};
pub use json_read::read_geo_json;
pub use json_write::write_geo_json;
pub use obj::{read_obj, write_obj};
pub use ply::{PlyFormat, read_ply, write_ply, write_ply_format};
pub use types::*;

/// Sniff the container: legacy `PGEOMETRY` text (GPD) vs JSON. binary-JSON `0x7f NSJb` (bjson); `scf1` blosc-chunked `.bgeo.sc` (decompress, then re-sniff); legacy binary `BgeoV` -> `read_geo_bgeo_legacy` (v2 layout; other versions `Unsupported`).
pub fn read_geo(bytes: &[u8]) -> Result<Detail, GeoError> {
    if bytes.starts_with(&blosc::MAGIC) {
        return read_geo_sc(bytes);
    }
    let head = bytes.iter().copied().find(|b| !b.is_ascii_whitespace());
    match head {
        Some(b'P') => read_geo_gpd(bytes),
        Some(b'[') | Some(b'{') => read_geo_json(bytes),
        Some(0x7f) => read_geo_bjson(bytes),
        Some(b'B') => read_geo_bgeo_legacy(bytes),
        _ => Err(GeoError::Syntax("unrecognised geometry container".into())),
    }
}
