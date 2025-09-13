// PhysicsControls.jsx
import React from 'react';

const PhysicsControls = ({ values, onChange, className = '' }) => {
  const handleSlider = (param, newValue) => {
    onChange({ ...values, [param]: Number(newValue) });
  };

  return (
    <div className={`${className}`}>
      <div className="flex items-center mb-3">
        <label className="w-24 text-sm font-medium text-gray-700">Repulsion</label>
        <input
          type="range" min="0" max="200" step="10"
          value={values.repulsion}
          onChange={(e) => handleSlider('repulsion', e.target.value)}
          className="flex-1 mr-3 accent-teal-600 cursor-pointer"
        />
        <span className="w-12 text-sm text-gray-700 text-right">{values.repulsion}</span>
      </div>
      <div className="flex items-center">
        <label className="w-24 text-sm font-medium text-gray-700">Link Length</label>
        <input
          type="range" min="50" max="300" step="10"
          value={values.linkDistance}
          onChange={(e) => handleSlider('linkDistance', e.target.value)}
          className="flex-1 mr-3 accent-teal-600 cursor-pointer"
        />
        <span className="w-12 text-sm text-gray-700 text-right">{values.linkDistance}</span>
      </div>
    </div>
  );
};

export default PhysicsControls;
