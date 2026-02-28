"use client";

import { useState, type ReactElement } from "react";
import clsx from "clsx";

type ZoomableImageProps = {
    src: string;
    alt: string;
};

export function ZoomableImage({ src, alt }: ZoomableImageProps): ReactElement {
    const [isZoomed, setIsZoomed] = useState(false);

    return (
        <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
                src={src}
                alt={alt}
                className={clsx("zoomable-image", { "zoomable-image--active": isZoomed })}
                onClick={() => setIsZoomed(true)}
                style={{ cursor: "zoom-in" }}
            />

            {isZoomed && (
                <div
                    className="image-lightbox-overlay"
                    onClick={() => setIsZoomed(false)}
                >
                    <div className="image-lightbox-content">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                            src={src}
                            alt={alt}
                            onClick={(e) => e.stopPropagation()}
                        />
                        <button
                            className="image-lightbox-close"
                            onClick={() => setIsZoomed(false)}
                            aria-label="Close image"
                        >
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
