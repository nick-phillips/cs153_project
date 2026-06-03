export default function EmptyState() {
  return (
    <div className="empty-state">
      <h2>Biomarker interpretation results</h2>
      <p className="muted">
        Select a compound from the list to view its report, or search by compound id,
        drug name, or gene/feature (e.g. <code>MDM4</code>).
      </p>
    </div>
  );
}
