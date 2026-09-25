// Shared by the existing human browser adapter's example presentations.
// Storage may be blocked by browser policy; input must keep working regardless.
export function safeStorage(name) {
  return {
    get(key) {
      try { return window[name].getItem(key); } catch { return null; }
    },
    set(key, value) {
      try { window[name].setItem(key, value); } catch {}
    },
    remove(key) {
      try { window[name].removeItem(key); } catch {}
    },
  };
}
