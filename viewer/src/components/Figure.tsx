import { useState } from 'react';

export default function Figure({ src, caption }: { src: string; caption?: string }) {
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState(false);
  if (err) return null;
  return (
    <figure className="figure">
      <img
        src={src}
        alt={caption ?? ''}
        loading="lazy"
        onClick={() => setOpen(true)}
        onError={() => setErr(true)}
      />
      {caption && <figcaption>{caption}</figcaption>}
      {open && (
        <div className="lightbox" onClick={() => setOpen(false)} role="presentation">
          <img src={src} alt={caption ?? ''} />
        </div>
      )}
    </figure>
  );
}
