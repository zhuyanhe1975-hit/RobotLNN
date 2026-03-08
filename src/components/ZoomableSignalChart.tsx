import React, { useEffect, useMemo, useRef, useState } from 'react';
import { extent, line as d3Line, scaleLinear, select, zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3';

type SeriesConfig<T extends object> = {
  key: keyof T & string;
  color: string;
  label: string;
  dashed?: boolean;
  opacity?: number;
};

type Props<T extends object> = {
  data: T[];
  xKey: keyof T & string;
  series: Array<SeriesConfig<T>>;
  yFormatter?: (value: number) => string;
  className?: string;
};

const margin = { top: 12, right: 16, bottom: 24, left: 48 };

function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => {
      const box = entry.contentRect;
      setSize({ width: box.width, height: box.height });
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return [ref, size] as const;
}

function sampleSeries<T extends object>(
  points: T[],
  pixelWidth: number
) {
  if (points.length <= Math.max(pixelWidth * 2, 400)) return points;
  const stride = Math.ceil(points.length / Math.max(pixelWidth * 2, 400));
  const reduced: T[] = [];
  for (let i = 0; i < points.length; i += stride) reduced.push(points[i]);
  if (reduced[reduced.length - 1] !== points[points.length - 1]) reduced.push(points[points.length - 1]);
  return reduced;
}

export function ZoomableSignalChart<T extends object>({
  data,
  xKey,
  series,
  yFormatter = (value) => value.toFixed(2),
  className,
}: Props<T>) {
  const [plotRef, size] = useElementSize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [transform, setTransform] = useState<ZoomTransform>(zoomIdentity);

  const width = size.width;
  const height = size.height;
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

  const xValues = useMemo(
    () => data.map((point) => Number((point as Record<string, unknown>)[xKey] ?? 0)).filter((value) => Number.isFinite(value)),
    [data, xKey]
  );

  const xDomain = useMemo<[number, number]>(() => {
    const domain = extent(xValues) as [number | undefined, number | undefined];
    if (!Number.isFinite(domain[0]) || !Number.isFinite(domain[1])) return [0, 1];
    if (domain[0] === domain[1]) return [domain[0] as number, (domain[1] as number) + 1];
    return [domain[0] as number, domain[1] as number];
  }, [xValues]);

  const baseXScale = useMemo(
    () => scaleLinear().domain(xDomain).range([0, innerWidth]),
    [xDomain, innerWidth]
  );

  const zoomedXScale = useMemo(() => transform.rescaleX(baseXScale), [transform, baseXScale]);
  const visibleDomain = zoomedXScale.domain();

  const visibleData = useMemo(() => {
    const minX = visibleDomain[0];
    const maxX = visibleDomain[1];
    const filtered = data.filter((point) => {
      const x = Number((point as Record<string, unknown>)[xKey] ?? 0);
      return x >= minX && x <= maxX;
    });
    const source = filtered.length > 0 ? filtered : data;
    const hasSeriesGaps = source.some((point) =>
      series.some((item) => {
        const value = Number((point as Record<string, unknown>)[item.key]);
        return !Number.isFinite(value);
      })
    );
    return hasSeriesGaps ? source : sampleSeries(source, innerWidth || 1);
  }, [data, innerWidth, series, visibleDomain, xKey]);

  const yDomain = useMemo<[number, number]>(() => {
    const values: number[] = [];
    for (const point of visibleData) {
      for (const item of series) {
        const value = Number((point as Record<string, unknown>)[item.key]);
        if (Number.isFinite(value)) values.push(value);
      }
    }
    if (values.length === 0) return [-1, 1];
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) return [min - 1, max + 1];
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad];
  }, [visibleData, series]);

  const yScale = useMemo(
    () => scaleLinear().domain(yDomain).range([innerHeight, 0]),
    [yDomain, innerHeight]
  );

  const xTicks = useMemo(() => zoomedXScale.ticks(6), [zoomedXScale]);
  const yTicks = useMemo(() => yScale.ticks(5), [yScale]);

  useEffect(() => {
    if (!svgRef.current || innerWidth <= 0 || innerHeight <= 0) return;
    const zoomBehavior: ZoomBehavior<SVGSVGElement, unknown> = zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 100])
      .translateExtent([[0, 0], [width, height]])
      .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]])
      .filter((event) => !event.ctrlKey || event.type === 'wheel')
      .on('zoom', (event) => {
        setTransform(event.transform);
      });

    const selection = select(svgRef.current);
    selection.call(zoomBehavior);
    return () => {
      selection.on('.zoom', null);
    };
  }, [height, innerHeight, innerWidth, width]);

  useEffect(() => {
    setTransform(zoomIdentity);
  }, [data, xKey]);

  return (
    <div className={className ?? 'h-full w-full flex flex-col'}>
      <div ref={plotRef} className="min-h-0 flex-1">
        <svg ref={svgRef} width={width} height={height} className="overflow-visible">
          <g transform={`translate(${margin.left},${margin.top})`}>
            <rect width={innerWidth} height={innerHeight} fill="transparent" />
            {yTicks.map((tick) => (
              <g key={`y-${tick}`} transform={`translate(0,${yScale(tick)})`}>
                <line x1={0} x2={innerWidth} y1={0} y2={0} stroke="#ffffff12" strokeDasharray="3 3" />
                <text x={-8} y={4} textAnchor="end" fill="#8e9299" fontSize="9">
                  {yFormatter(tick)}
                </text>
              </g>
            ))}
            {xTicks.map((tick) => (
              <g key={`x-${tick}`} transform={`translate(${zoomedXScale(tick)},0)`}>
                <line x1={0} x2={0} y1={0} y2={innerHeight} stroke="#ffffff08" />
                <text x={0} y={innerHeight + 16} textAnchor="middle" fill="#8e9299" fontSize="9">
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}
            {series.map((item) => {
              const pathData = d3Line<T>()
                .defined((point) => Number.isFinite(Number((point as Record<string, unknown>)[item.key])) && Number.isFinite(Number((point as Record<string, unknown>)[xKey])))
                .x((point) => zoomedXScale(Number((point as Record<string, unknown>)[xKey])))
                .y((point) => yScale(Number((point as Record<string, unknown>)[item.key])))(visibleData);

              return pathData ? (
                <path
                  key={item.key}
                  d={pathData}
                  fill="none"
                  stroke={item.color}
                  strokeWidth={1.75}
                  strokeDasharray={item.dashed ? '4 4' : undefined}
                  opacity={item.opacity ?? 1}
                />
              ) : null;
            })}
          </g>
        </svg>
      </div>
      <div className="mt-2 flex items-center justify-between text-[9px] font-mono text-[#8e9299]">
        <span>Wheel to zoom, drag to pan</span>
        <span>
          x-range {visibleDomain[0].toFixed(2)} - {visibleDomain[1].toFixed(2)}
        </span>
      </div>
    </div>
  );
}
