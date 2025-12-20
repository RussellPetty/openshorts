import React, { useState } from 'react';
import { Type, Palette } from 'lucide-react';

const CAPTION_STYLES = [
  {
    id: 'none',
    name: 'No Captions',
    preview: null,
    description: 'Video without captions'
  },
  {
    id: 'classic',
    name: 'Classic',
    preview: { color: '#FFFFFF', outline: '#000000', bg: null },
    description: 'White text with black outline'
  },
  {
    id: 'boxed',
    name: 'Boxed',
    preview: { color: '#FFFFFF', outline: null, bg: 'rgba(0,0,0,0.7)' },
    description: 'White text on dark background'
  },
  {
    id: 'yellow',
    name: 'Cinema',
    preview: { color: '#FFFF00', outline: '#000000', bg: null },
    description: 'Yellow movie-style subtitles'
  },
  {
    id: 'minimal',
    name: 'Minimal',
    preview: { color: '#FFFFFF', outline: null, bg: null, thin: true },
    description: 'Clean lowercase text'
  },
  {
    id: 'bold',
    name: 'Bold Impact',
    preview: { color: '#FFFFFF', outline: '#000000', bg: null, bold: true },
    description: 'Large bold text with thick outline'
  },
  {
    id: 'karaoke',
    name: 'Karaoke',
    preview: { color: '#FFFFFF', highlight: '#FFFF00', outline: '#000000' },
    description: 'Word-by-word highlight effect'
  },
  {
    id: 'neon',
    name: 'Neon Glow',
    preview: { color: '#FF00FF', outline: '#FF66FF', glow: true },
    description: 'Glowing neon effect'
  },
  {
    id: 'gradient',
    name: 'Gradient',
    preview: { gradient: ['#FF6666', '#6666FF'], outline: '#000000' },
    description: 'Colorful gradient text'
  }
];

const StylePreview = ({ style, isSelected, onClick }) => {
  const preview = style.preview;

  return (
    <button
      onClick={onClick}
      className={`relative p-3 rounded-lg border-2 transition-all duration-200 text-left w-full
        ${isSelected
          ? 'border-primary bg-primary/10 ring-2 ring-primary/30'
          : 'border-white/10 bg-white/5 hover:border-white/20 hover:bg-white/10'
        }`}
    >
      {/* Preview area */}
      <div className="h-12 flex items-center justify-center mb-2 rounded bg-gradient-to-br from-gray-800 to-gray-900 overflow-hidden">
        {!preview ? (
          <span className="text-gray-500 text-xs">No captions</span>
        ) : (
          <div
            className="px-2 py-1 rounded text-sm font-medium"
            style={{
              color: preview.gradient ? 'transparent' : preview.color,
              background: preview.gradient
                ? `linear-gradient(90deg, ${preview.gradient[0]}, ${preview.gradient[1]})`
                : preview.bg || 'transparent',
              backgroundClip: preview.gradient ? 'text' : undefined,
              WebkitBackgroundClip: preview.gradient ? 'text' : undefined,
              textShadow: preview.glow
                ? `0 0 10px ${preview.color}, 0 0 20px ${preview.color}`
                : preview.outline
                  ? `
                    -1px -1px 0 ${preview.outline},
                    1px -1px 0 ${preview.outline},
                    -1px 1px 0 ${preview.outline},
                    1px 1px 0 ${preview.outline}
                  `
                  : 'none',
              fontWeight: preview.bold ? 'bold' : preview.thin ? '300' : 'normal',
              textTransform: preview.thin ? 'lowercase' : 'none',
              fontSize: preview.bold ? '16px' : '14px'
            }}
          >
            {style.id === 'karaoke' ? (
              <>
                <span style={{ color: preview.color }}>Sample </span>
                <span style={{ color: preview.highlight }}>Text</span>
              </>
            ) : (
              'Sample Text'
            )}
          </div>
        )}
      </div>

      {/* Style name */}
      <div className="text-sm font-medium text-white">{style.name}</div>
      <div className="text-xs text-gray-400 truncate">{style.description}</div>

      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-primary" />
      )}
    </button>
  );
};

const CaptionStylePicker = ({
  selectedStyle,
  onStyleChange,
  customColor,
  onColorChange,
  customOutlineColor,
  onOutlineColorChange
}) => {
  const [showColorPicker, setShowColorPicker] = useState(false);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Type size={18} className="text-primary" />
          <h3 className="text-sm font-semibold text-white">Caption Style</h3>
        </div>
        {selectedStyle && selectedStyle !== 'none' && (
          <button
            onClick={() => setShowColorPicker(!showColorPicker)}
            className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-colors
              ${showColorPicker ? 'bg-primary/20 text-primary' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}
          >
            <Palette size={14} />
            Custom Colors
          </button>
        )}
      </div>

      {/* Style Grid */}
      <div className="grid grid-cols-3 gap-2">
        {CAPTION_STYLES.map((style) => (
          <StylePreview
            key={style.id}
            style={style}
            isSelected={selectedStyle === style.id}
            onClick={() => onStyleChange(style.id)}
          />
        ))}
      </div>

      {/* Color Customization */}
      {showColorPicker && selectedStyle && selectedStyle !== 'none' && (
        <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-3">
          <div className="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Custom Colors (Optional)
          </div>

          <div className="grid grid-cols-2 gap-3">
            {/* Text Color */}
            <div className="space-y-1.5">
              <label className="text-xs text-gray-300">Text Color</label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={customColor || '#FFFFFF'}
                  onChange={(e) => onColorChange(e.target.value)}
                  className="w-8 h-8 rounded cursor-pointer bg-transparent border border-white/20"
                />
                <input
                  type="text"
                  value={customColor || ''}
                  onChange={(e) => onColorChange(e.target.value)}
                  placeholder="#FFFFFF"
                  className="flex-1 px-2 py-1.5 rounded bg-white/5 border border-white/10 text-white text-xs
                    placeholder:text-gray-500 focus:outline-none focus:border-primary/50"
                />
              </div>
            </div>

            {/* Outline Color */}
            <div className="space-y-1.5">
              <label className="text-xs text-gray-300">Outline Color</label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={customOutlineColor || '#000000'}
                  onChange={(e) => onOutlineColorChange(e.target.value)}
                  className="w-8 h-8 rounded cursor-pointer bg-transparent border border-white/20"
                />
                <input
                  type="text"
                  value={customOutlineColor || ''}
                  onChange={(e) => onOutlineColorChange(e.target.value)}
                  placeholder="#000000"
                  className="flex-1 px-2 py-1.5 rounded bg-white/5 border border-white/10 text-white text-xs
                    placeholder:text-gray-500 focus:outline-none focus:border-primary/50"
                />
              </div>
            </div>
          </div>

          {/* Clear button */}
          {(customColor || customOutlineColor) && (
            <button
              onClick={() => {
                onColorChange(null);
                onOutlineColorChange(null);
              }}
              className="text-xs text-gray-400 hover:text-white transition-colors"
            >
              Reset to default colors
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default CaptionStylePicker;
