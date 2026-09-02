// Navbar: brand + project + worker status (connect/save wiring is a later milestone).
export function Navbar() {
  return (
    <header className="nav">
      <div className="brand">
        <span className="mark" />
        <span className="brand-name">Yapuny</span>
        <span className="proj">
          model <b>untitled</b>
        </span>
      </div>
      <div className="nav-r">
        <div className="tele">
          <span className="d off" />
          worker: —
        </div>
        {/* TODO: Connect worker (URL + optional token), save/load, settings */}
        <button className="nl" type="button">
          connect
        </button>
      </div>
    </header>
  );
}
