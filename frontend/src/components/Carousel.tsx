import { Children, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface Props {
  children: ReactNode;
  label: string;
  /** Card width as a CSS length; the carousel shows as many as fit. */
  itemWidth?: string;
  className?: string;
}

/**
 * Horizontal card carousel: native scroll with snap points (touch, trackpad and keyboard all
 * work), plus previous/next buttons that appear only when there is more to see.
 */
export default function Carousel({ children, label, itemWidth = "280px", className = "" }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: true, end: true });
  const items = Children.toArray(children);

  const update = useCallback(() => {
    const el = track.current;
    if (!el) return;
    setEdges({ start: el.scrollLeft <= 4, end: el.scrollLeft + el.clientWidth >= el.scrollWidth - 4 });
  }, []);

  useEffect(() => {
    update();
    const el = track.current;
    if (!el) return;
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [update, items.length]);

  const page = (direction: 1 | -1) => {
    const el = track.current;
    if (el) el.scrollBy({ left: direction * el.clientWidth * 0.85, behavior: "smooth" });
  };

  if (!items.length) return null;
  return (
    <div className={`carousel ${className}`} role="region" aria-roledescription="carousel" aria-label={label}>
      <div
        className="carousel-track"
        ref={track}
        onScroll={update}
        tabIndex={0}
        style={{ ["--item-w" as string]: itemWidth }}
      >
        {items.map((child, i) => (
          <div className="carousel-item" key={i} role="group" aria-roledescription="slide" aria-label={`${i + 1} of ${items.length}`}>
            {child}
          </div>
        ))}
      </div>
      {!edges.start && (
        <button className="carousel-nav prev" onClick={() => page(-1)} aria-label="Previous">
          <ChevronLeft size={16} />
        </button>
      )}
      {!edges.end && (
        <button className="carousel-nav next" onClick={() => page(1)} aria-label="Next">
          <ChevronRight size={16} />
        </button>
      )}
    </div>
  );
}
