"use client";

import { useRouter } from "next/navigation";
import { useEffect, useId, useState, type ReactElement } from "react";
import { createPortal } from "react-dom";
import { MermaidBlock } from "@/components/markdown/MermaidBlock";

type MarkdownRendererProps = {
  html: string;
};

function isModifiedClick(event: MouseEvent): boolean {
  return event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
}

function shouldUseSpaNavigation(href: string): boolean {
  return href.startsWith("/") || href.startsWith("./") || href.startsWith("../");
}

function resolveHrefToPath(href: string): string {
  const resolved = new URL(href, window.location.href);
  return `${resolved.pathname}${resolved.search}${resolved.hash}`;
}

export function MarkdownRenderer({ html }: MarkdownRendererProps): ReactElement {
  const rawId = useId();
  const containerId = `markdown-${rawId.replace(/:/g, "")}`;
  const router = useRouter();

  const [zoomedImage, setZoomedImage] = useState<{ src: string; alt: string } | null>(null);

  useEffect(() => {
    const container = document.getElementById(containerId);
    if (!container) {
      return;
    }

    const clickHandler = async (event: MouseEvent): Promise<void> => {
      const target = event.target as HTMLElement | null;
      if (!target) {
        return;
      }

      const copyButton = target.closest(".copy-code-btn") as HTMLButtonElement | null;
      if (copyButton) {
        const pre = copyButton.closest("pre");
        const code = pre?.querySelector("code");
        const payload = code?.textContent ?? "";
        if (!payload) {
          return;
        }
        try {
          await navigator.clipboard.writeText(payload);
          copyButton.textContent = "Copied";
          window.setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1200);
        } catch {
          copyButton.textContent = "Error";
          window.setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1200);
        }
        event.preventDefault();
        return;
      }

      const img = target.closest("img") as HTMLImageElement | null;
      if (img && !img.closest(".image-lightbox-content")) {
        setZoomedImage({ src: img.src, alt: img.alt || "Zoomed image" });
        return;
      }

      const anchor = target.closest("a[href]") as HTMLAnchorElement | null;
      if (!anchor) {
        return;
      }
      if (event.defaultPrevented || isModifiedClick(event) || event.button !== 0) {
        return;
      }

      const href = anchor.getAttribute("href") ?? "";
      if (!href || href.startsWith("#")) {
        return;
      }
      if (!shouldUseSpaNavigation(href)) {
        return;
      }

      event.preventDefault();
      router.push(resolveHrefToPath(href));
    };

    container.addEventListener("click", clickHandler);

    return () => {
      container.removeEventListener("click", clickHandler);
    };
  }, [containerId, router, html]);

  return (
    <>
      <section className="markdown-host">
        <div id={containerId} className="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />
        <MermaidBlock containerId={containerId} />
      </section>

      {zoomedImage && typeof document !== "undefined" && createPortal(
        <div
          className="image-lightbox-overlay"
          onClick={() => setZoomedImage(null)}
        >
          <div className="image-lightbox-content">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={zoomedImage.src}
              alt={zoomedImage.alt}
              onClick={(e) => e.stopPropagation()}
            />
            <button
              className="image-lightbox-close"
              onClick={() => setZoomedImage(null)}
              aria-label="Close image"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 16, height: 16 }}>
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
