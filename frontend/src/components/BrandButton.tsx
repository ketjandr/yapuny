// The clickable Yapuny brand mark; returns to the models home. Shared by the app navbar and the
// models page so the logo behaves identically everywhere.
import { useNavigate } from "react-router-dom";

export function BrandButton() {
  const navigate = useNavigate();
  return (
    <button className="brand-btn" type="button" onClick={() => navigate("/")} title="Back to models">
      <span className="mark" />
      <span className="brand-name">Yapuny</span>
    </button>
  );
}
