import React from 'react';
import { motion } from 'motion/react';

interface RobotAvatarProps {
  isThinking?: boolean;
  isSpeaking?: boolean;
}

export const RobotAvatar: React.FC<RobotAvatarProps> = ({ isThinking, isSpeaking }) => {
  return (
    <div className="relative w-48 h-48 flex items-center justify-center">
      {/* Background Glow */}
      <motion.div
        animate={{
          scale: isThinking ? [1, 1.1, 1] : 1,
          opacity: isThinking ? [0.2, 0.4, 0.2] : 0.2,
        }}
        transition={{ duration: 2, repeat: Infinity }}
        className="absolute inset-0 bg-[#00ff9d] rounded-full blur-3xl"
      />

      <svg viewBox="0 0 200 200" className="w-full h-full relative z-10">
        {/* Head Base */}
        <rect
          x="50"
          y="40"
          width="100"
          height="120"
          rx="20"
          fill="#1a1b1e"
          stroke="#00ff9d"
          strokeWidth="2"
        />
        
        {/* Antenna */}
        <line x1="100" y1="40" x2="100" y2="20" stroke="#00ff9d" strokeWidth="2" />
        <motion.circle
          cx="100"
          cy="20"
          r="4"
          fill="#00ff9d"
          animate={{
            opacity: isThinking ? [0.3, 1, 0.3] : 1,
            scale: isThinking ? [1, 1.5, 1] : 1,
          }}
          transition={{ duration: 1, repeat: Infinity }}
        />

        {/* Eyes Area */}
        <rect x="65" y="70" width="70" height="30" rx="5" fill="#0a0a0b" />
        
        {/* Left Eye */}
        <motion.circle
          cx="80"
          cy="85"
          r="6"
          fill="#00ff9d"
          animate={{
            scaleY: isThinking ? [1, 0.1, 1] : 1,
          }}
          transition={{ duration: 3, repeat: Infinity, repeatDelay: 2 }}
        />
        
        {/* Right Eye */}
        <motion.circle
          cx="120"
          cy="85"
          r="6"
          fill="#00ff9d"
          animate={{
            scaleY: isThinking ? [1, 0.1, 1] : 1,
          }}
          transition={{ duration: 3, repeat: Infinity, repeatDelay: 2.1 }}
        />

        {/* Mouth / Speaker */}
        <motion.rect
          x="80"
          y="120"
          width="40"
          height="4"
          rx="2"
          fill="#00ff9d"
          animate={{
            height: isSpeaking ? [4, 12, 4] : 4,
            y: isSpeaking ? [120, 116, 120] : 120,
          }}
          transition={{ duration: 0.2, repeat: Infinity }}
        />

        {/* Decorative Lines */}
        <line x1="50" y1="130" x2="150" y2="130" stroke="#00ff9d" strokeWidth="0.5" strokeOpacity="0.3" />
        <line x1="50" y1="140" x2="150" y2="140" stroke="#00ff9d" strokeWidth="0.5" strokeOpacity="0.3" />
      </svg>
    </div>
  );
};
