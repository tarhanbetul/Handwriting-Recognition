// src/components/PredictionResult.js
import React from 'react';

const PredictionResult = ({ image, prediction }) => {
  return (
    <div>
      <img src={image} alt="Uploaded" />
      <p>Tahmin Edilen Sayı: {prediction}</p>
    </div>
  );
};

export default PredictionResult;
