//! File browser backend (File > Open dialog, `ui/src/filewin.ts`): directory listing, standard places, new folder.
//! std only; paths travel as `/`-separated strings (Windows accepts them), directories end in `/`.
//!
//! SANDBOX: every path the UI sends is resolved inside the project root (`HEXE_PROJECT_ROOT`, else the process
//! working directory). `..`, absolute paths outside the root and symlinks that leave it are errors, not listings.
use serde::Serialize;
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Listing cap: a hostile / huge directory cannot stall the UI with millions of rows.
pub const MAX_ENTRIES: usize = 20_000;

/// One directory entry as the browser shows it.
#[derive(Serialize, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub dir: bool,
    /// last modified, ms since the Unix epoch (None = unreadable)
    pub modified: Option<f64>,
    /// bytes (None for directories / unreadable)
    pub size: Option<u64>,
}

/// Locations column: home, desktop, temp, filesystem roots (`C:/`, ... on Windows; `/` elsewhere).
#[derive(Serialize, Debug)]
pub struct Places {
    pub home: String,
    pub desktop: String,
    pub temp: String,
    pub roots: Vec<String>,
}

/// `C:\Users\x` -> `C:/Users/x/` (directory form the UI works in).
pub fn dir_string(p: &Path) -> String {
    let mut s = p.to_string_lossy().replace('\\', "/");
    if !s.ends_with('/') {
        s.push('/');
    }
    s
}

/// Project root: `HEXE_PROJECT_ROOT`, else the working directory (canonical when it resolves).
pub fn root() -> PathBuf {
    let p = std::env::var_os("HEXE_PROJECT_ROOT")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("/"));
    p.canonicalize().unwrap_or(p)
}

/// `dir` as a path inside `root`, or an error. Empty / `..` / absolute-outside / symlink-out are all rejected.
pub fn under(root: &Path, dir: &str) -> Result<PathBuf, String> {
    let raw = dir.replace('\\', "/");
    if raw.trim().is_empty() {
        return Err("empty path".to_string());
    }
    let p = Path::new(&raw);
    if p.components().any(|c| c == Component::ParentDir) {
        return Err(format!("{dir}: `..` is not allowed"));
    }
    let joined = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
    let real = joined.canonicalize().unwrap_or(joined);
    let base = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if !real.starts_with(&base) {
        return Err(format!("{dir}: outside the project root {}", base.display()));
    }
    Ok(real)
}

fn home() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Dot names everywhere (Houdini hides `.android`, `.vscode`, ... on Windows too), plus hidden / system attributes.
#[cfg(windows)]
fn hidden(name: &str, meta: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const HIDDEN: u32 = 0x2;
    const SYSTEM: u32 = 0x4;
    name.starts_with('.') || meta.file_attributes() & (HIDDEN | SYSTEM) != 0
}
#[cfg(not(windows))]
fn hidden(name: &str, _meta: &std::fs::Metadata) -> bool {
    name.starts_with('.')
}

/// Entries of `dir` inside the project root (`under`), hidden / system ones skipped, at most `MAX_ENTRIES`.
pub fn list(dir: &str) -> Result<Vec<Entry>, String> {
    list_in(&root(), dir)
}

/// `list` against an explicit root (the tests' entry point).
pub fn list_in(root: &Path, dir: &str) -> Result<Vec<Entry>, String> {
    let path = under(root, dir)?;
    let rd = std::fs::read_dir(&path).map_err(|e| format!("{dir}: {e}"))?;
    let mut out = Vec::new();
    for ent in rd.flatten() {
        if out.len() >= MAX_ENTRIES {
            break;
        }
        let name = ent.file_name().to_string_lossy().into_owned();
        // follow links (a linked folder browses as a folder); a dangling link falls back to the link itself
        let Ok(meta) = std::fs::metadata(ent.path()).or_else(|_| ent.metadata()) else { continue };
        if hidden(&name, &meta) {
            continue;
        }
        let modified =
            meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_millis() as f64);
        let dir = meta.is_dir();
        out.push(Entry { name, dir, modified, size: if dir { None } else { Some(meta.len()) } });
    }
    Ok(out)
}

pub fn places() -> Places {
    places_in(&root())
}

/// Places clamped to `root`: anything outside it (home, Desktop, temp, drive roots) collapses to the root itself.
pub fn places_in(root: &Path) -> Places {
    let base = dir_string(root);
    let keep = |p: PathBuf| match under(root, &dir_string(&p)) {
        Ok(real) if real.is_dir() => dir_string(&real),
        _ => base.clone(),
    };
    let h = home();
    Places {
        home: keep(h.clone()),
        desktop: keep(h.join("Desktop")),
        temp: keep(std::env::temp_dir()),
        roots: vec![base],
    }
}

/// New Folder: `dir/New Folder`, or `New Folder 2`, ... when taken. Returns the created name.
pub fn mkdir(dir: &str) -> Result<String, String> {
    mkdir_in(&root(), dir)
}

/// `mkdir` against an explicit root (the tests' entry point).
pub fn mkdir_in(root: &Path, dir: &str) -> Result<String, String> {
    let base = under(root, dir)?;
    for i in 1..1000 {
        let name = if i == 1 { "New Folder".to_string() } else { format!("New Folder {i}") };
        let p = base.join(&name);
        if p.exists() {
            continue;
        }
        return std::fs::create_dir(&p).map(|()| name).map_err(|e| format!("{}: {e}", p.display()));
    }
    Err(format!("{dir}: no free New Folder name"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch root per test (temp dir; canonical so macOS `/var` -> `/private/var` does not fail prefix checks).
    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("hexe-files-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("scratch");
        p.canonicalize().unwrap_or(p)
    }

    #[test]
    fn list_mkdir_round_trip_and_hostile_dirs() -> Result<(), String> {
        let base = scratch("roundtrip");
        std::fs::write(base.join("a.hip"), b"12345").map_err(|e| e.to_string())?;
        std::fs::create_dir_all(base.join(".hidden")).map_err(|e| e.to_string())?;
        let b = dir_string(&base);
        assert_eq!(mkdir_in(&base, &b)?, "New Folder");
        assert_eq!(mkdir_in(&base, &b)?, "New Folder 2");
        let mut l = list_in(&base, &b)?;
        l.sort_by(|x, y| x.name.cmp(&y.name));
        assert_eq!(
            l.iter().map(|e| (e.name.as_str(), e.dir, e.size)).collect::<Vec<_>>(),
            [("New Folder", true, None), ("New Folder 2", true, None), ("a.hip", false, Some(5))]
        );
        assert!(l.iter().all(|e| e.modified.is_some()));
        assert!(list_in(&base, &format!("{b}does-not-exist/")).is_err());
        assert!(list_in(&base, "").is_err());
        // relative paths are resolved against the root
        assert_eq!(list_in(&base, "New Folder")?.len(), 0);
        std::fs::remove_dir_all(&base).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// SANDBOX: `..`, absolute paths outside the root and symlinks out of it are errors; inside stays allowed.
    #[test]
    fn paths_outside_the_project_root_are_rejected() -> Result<(), String> {
        let base = scratch("sandbox");
        let root = base.join("proj");
        std::fs::create_dir_all(root.join("sub")).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(base.join("secret")).map_err(|e| e.to_string())?;
        std::fs::write(base.join("secret/key.txt"), b"x").map_err(|e| e.to_string())?;

        assert!(under(&root, "sub").is_ok());
        assert!(under(&root, &dir_string(&root.join("sub"))).is_ok());
        for bad in [
            "..".to_string(),
            "../secret".to_string(),
            "sub/../../secret".to_string(),
            "sub\\..\\..\\secret".to_string(),
            dir_string(&base.join("secret")),
            "/".to_string(),
            "/etc".to_string(),
            String::new(),
            "   ".to_string(),
        ] {
            assert!(under(&root, &bad).is_err(), "{bad:?} must be rejected");
            assert!(list_in(&root, &bad).is_err(), "list {bad:?} must be rejected");
            assert!(mkdir_in(&root, &bad).is_err(), "mkdir {bad:?} must be rejected");
        }
        assert!(list_in(&root, &dir_string(&base.join("secret"))).unwrap_err().contains("outside the project root"));

        // a symlink inside the root pointing out of it is still out
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("secret"), root.join("escape")).map_err(|e| e.to_string())?;
            assert!(under(&root, "escape").is_err());
            assert!(list_in(&root, "escape").is_err());
        }

        // mkdir only writes inside the root
        assert_eq!(mkdir_in(&root, "sub")?, "New Folder");
        assert!(root.join("sub/New Folder").is_dir());
        assert!(!base.join("secret/New Folder").exists());

        // places collapse to the root when home / temp lie outside it
        let p = places_in(&root);
        let want = dir_string(&root);
        assert_eq!((p.home, p.desktop, p.temp, p.roots), (want.clone(), want.clone(), want.clone(), vec![want]));

        std::fs::remove_dir_all(&base).map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn dir_string_forward_slashes_and_trailing_slash() {
        assert_eq!(dir_string(Path::new("C:\\Users\\x")), "C:/Users/x/");
        assert_eq!(dir_string(Path::new("/tmp/")), "/tmp/");
        let p = places();
        assert!(p.home.ends_with('/') && p.temp.ends_with('/') && !p.roots.is_empty());
        let base = dir_string(&root());
        assert!([&p.home, &p.desktop, &p.temp].into_iter().chain(&p.roots).all(|x| x.starts_with(&base)));
    }
}
