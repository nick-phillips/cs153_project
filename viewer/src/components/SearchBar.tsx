export default function SearchBar({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="searchbar">
      <input
        type="search"
        placeholder="Search compound id, drug name, or gene/feature (e.g. MDM4)…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoFocus
      />
    </div>
  );
}
