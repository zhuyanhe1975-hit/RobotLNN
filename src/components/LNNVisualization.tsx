import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface Node extends d3.SimulationNodeDatum {
  id: number;
  group: number;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  value: number;
}

export const LNNVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [angle, setAngle] = useState(0);

  useEffect(() => {
    let frameId: number;
    const animate = (time: number) => {
      // Simulate a 1-DOF joint oscillation
      setAngle(Math.sin(time / 1000) * 45);
      frameId = requestAnimationFrame(animate);
    };
    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, []);

  useEffect(() => {
    if (!svgRef.current) return;
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);

    // Base
    g.append('circle')
      .attr('r', 10)
      .attr('fill', '#1a1b1e')
      .attr('stroke', '#00ff9d')
      .attr('stroke-width', 2);

    // Arm
    const arm = g.append('g')
      .attr('transform', `rotate(${angle})`);

    arm.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', 60)
      .attr('y2', 0)
      .attr('stroke', '#00ff9d')
      .attr('stroke-width', 4)
      .attr('stroke-linecap', 'round');

    arm.append('circle')
      .attr('cx', 60)
      .attr('cy', 0)
      .attr('r', 6)
      .attr('fill', '#00ff9d')
      .attr('filter', 'blur(1px)');

    // Data points / Neural activity
    const nodes = Array.from({ length: 12 }, (_, i) => ({
      x: (Math.random() - 0.5) * width,
      y: (Math.random() - 0.5) * height,
    }));

    svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('cx', d => d.x + width / 2)
      .attr('cy', d => d.y + height / 2)
      .attr('r', 1)
      .attr('fill', '#00ff9d')
      .attr('opacity', 0.3);

  }, [angle]);

  return (
    <div className="w-full h-full relative overflow-hidden bg-black/20 rounded-lg hardware-border">
      <div className="absolute top-2 left-3 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-[#00ff9d] animate-pulse" />
        <span className="text-[10px] font-mono text-[#00ff9d]/70 uppercase tracking-widest">1-DOF Joint Dynamics Simulation</span>
      </div>
      <div className="absolute bottom-2 right-3 text-[9px] font-mono text-[#00ff9d]/50">
        θ: {angle.toFixed(2)}°
      </div>
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};
