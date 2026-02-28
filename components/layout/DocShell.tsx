"use client";

import { useMemo, useState, useEffect, useCallback, type ReactElement } from "react";
import clsx from "clsx";
import type {
  DocRecord,
  NavigationData,
  RelatedDoc,
  SearchIndex,
} from "@/content/types";
import { LeftRail } from "@/components/layout/LeftRail";
import { RightRail } from "@/components/layout/RightRail";
import { MarkdownRenderer } from "@/components/markdown/MarkdownRenderer";

type DocShellProps = {
  doc: DocRecord;
  navigation: NavigationData;
  searchIndex: SearchIndex;
  relatedDocs: RelatedDoc[];
};

function normalizeHeaderTitle(title: string): string {
  const normalized = title
    .replace(/^chapter\s+\d+(?:\.\d+)*\s*:\s*/i, "")
    .trim();
  return normalized.length > 0 ? normalized : title;
}

function RailToggleIcon({
  side,
  collapsed,
}: {
  side: "left" | "right";
  collapsed: boolean;
}): ReactElement {
  const arrowDirection =
    side === "left"
      ? collapsed
        ? "right"
        : "left"
      : collapsed
        ? "left"
        : "right";
  const arrowPath =
    arrowDirection === "left" ? "M12 6 8 10l4 4" : "M8 6l4 4-4 4";
  const dividerPath = side === "left" ? "M7 4.5v11" : "M13 4.5v11";

  return (
    <svg
      className="doc-shell__toggle-icon"
      viewBox="0 0 20 20"
      aria-hidden="true"
      focusable="false"
    >
      <rect
        className="doc-shell__toggle-icon-frame"
        x="2.5"
        y="3"
        width="15"
        height="14"
        rx="3"
      />
      <path className="doc-shell__toggle-icon-divider" d={dividerPath} />
      <path className="doc-shell__toggle-icon-arrow" d={arrowPath} />
    </svg>
  );
}

export function DocShell({
  doc,
  navigation,
  searchIndex,
  relatedDocs,
}: DocShellProps): ReactElement {
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);
  const [leftDesktopVisible, setLeftDesktopVisible] = useState(true);
  const [rightDesktopVisible, setRightDesktopVisible] = useState(true);

  // Resize state
  const [leftWidth, setLeftWidth] = useState(260); // Default left sidebar width
  const [rightWidth, setRightWidth] = useState(240); // Default right sidebar width
  const [isDraggingLeft, setIsDraggingLeft] = useState(false);
  const [isDraggingRight, setIsDraggingRight] = useState(false);

  const activePath = useMemo(() => doc.canonicalPath, [doc.canonicalPath]);
  const headerTitle = useMemo(
    () => normalizeHeaderTitle(doc.title),
    [doc.title],
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDraggingLeft) {
        // Limit width between 200px and 600px
        const newWidth = Math.max(200, Math.min(600, e.clientX));
        setLeftWidth(newWidth);
      } else if (isDraggingRight) {
        // Limit width between 200px and 600px from the right edge
        const newWidth = Math.max(200, Math.min(600, window.innerWidth - e.clientX));
        setRightWidth(newWidth);
      }
    },
    [isDraggingLeft, isDraggingRight]
  );

  const handleMouseUp = useCallback(() => {
    setIsDraggingLeft(false);
    setIsDraggingRight(false);
  }, []);

  useEffect(() => {
    if (isDraggingLeft || isDraggingRight) {
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    } else {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDraggingLeft, isDraggingRight, handleMouseMove, handleMouseUp]);

  return (
    <div
      className="doc-shell"
      style={
        {
          "--left-nav-width": leftDesktopVisible ? `${leftWidth}px` : "0px",
          "--right-nav-width": rightDesktopVisible ? `${rightWidth}px` : "0px",
        } as React.CSSProperties
      }
    >
      <header className="doc-shell__header" role="banner">
        <div className="doc-shell__header-side doc-shell__header-side--left">
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--mobile"
            aria-expanded={leftOpen}
            aria-controls="docs-left-rail"
            onClick={() => setLeftOpen((open) => !open)}
          >
            Contents
          </button>
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--desktop"
            aria-label={
              leftDesktopVisible
                ? "Collapse chapters rail"
                : "Expand chapters rail"
            }
            aria-pressed={!leftDesktopVisible}
            onClick={() => setLeftDesktopVisible((visible) => !visible)}
          >
            <RailToggleIcon side="left" collapsed={!leftDesktopVisible} />
          </button>
        </div>
        <div className="doc-shell__title-wrap">
          <p className="doc-shell__eyebrow">Agentic AI Knowledge Platform</p>
          <h1 className="doc-shell__title" id="doc-title">
            {headerTitle}
          </h1>
        </div>
        <div className="doc-shell__header-side doc-shell__header-side--right">
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--mobile"
            aria-expanded={rightOpen}
            aria-controls="docs-right-rail"
            onClick={() => setRightOpen((open) => !open)}
          >
            Search
          </button>
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--desktop"
            aria-label={
              rightDesktopVisible ? "Collapse tools rail" : "Expand tools rail"
            }
            aria-pressed={!rightDesktopVisible}
            onClick={() => setRightDesktopVisible((visible) => !visible)}
          >
            <RailToggleIcon side="right" collapsed={!rightDesktopVisible} />
          </button>
        </div>
      </header>

      <div
        className={clsx("doc-shell__layout", {
          "doc-shell__layout--left-collapsed": !leftDesktopVisible,
          "doc-shell__layout--right-collapsed": !rightDesktopVisible,
          "doc-shell__layout--is-dragging": isDraggingLeft || isDraggingRight,
        })}
      >
        <div className="doc-shell__left-container">
          <LeftRail
            railId="docs-left-rail"
            navigation={navigation}
            activePath={activePath}
            mobileOpen={leftOpen}
            onClose={() => setLeftOpen(false)}
          />
          {leftDesktopVisible && (
            <div
              className="doc-shell__splitter doc-shell__splitter--left"
              onMouseDown={() => setIsDraggingLeft(true)}
              aria-hidden="true"
            >
              <div className="doc-shell__splitter-handle">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 16, height: 16, color: "#94a3b8" }}>
                  <path d="M11 5l-7 7 7 7M13 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          )}
        </div>

        <main
          className="doc-shell__main"
          role="main"
          aria-labelledby="doc-title"
        >
          <article className="doc-shell__article" aria-label="Document content">
            <p className="doc-shell__meta">
              <span>{doc.readingMinutes} min read</span>
              <span>{doc.sourcePath}</span>
            </p>
            <MarkdownRenderer html={doc.html} />
          </article>
        </main>

        <div className="doc-shell__right-container">
          {rightDesktopVisible && (
            <div
              className="doc-shell__splitter doc-shell__splitter--right"
              onMouseDown={() => setIsDraggingRight(true)}
              aria-hidden="true"
            >
              <div className="doc-shell__splitter-handle">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 16, height: 16, color: "#94a3b8" }}>
                  <path d="M11 5l-7 7 7 7M13 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          )}
          <div
            id="docs-right-rail"
            className={clsx("doc-shell__right-wrap", {
              "doc-shell__right-wrap--open": rightOpen,
            })}
          >
            <RightRail
              searchIndex={searchIndex}
              headings={doc.headings}
              relatedDocs={relatedDocs}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

