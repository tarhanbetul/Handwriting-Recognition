import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setResult('');
  };

  const handleUpload = async () => {
    try {
      if (!selectedFile) {
        setResult('Please select a file.');
        return;
      }
       // Dosya türünü kontrol et (PNG veya JPEG olmalı)
      if (selectedFile.type !== 'image/png' && selectedFile.type !== 'image/jpeg') {
      setResult('Please upload a valid PNG or JPEG file.');
      return;
    }
      const formData = new FormData();
      formData.append('file', selectedFile, selectedFile.name);
      const url = process.env.REACT_APP_API_URL;
      const data = formData;

      try {
        const response = await axios.post(url, data);

        if (response.status === 200) {
          const responseData = response.data;
          setResult('Tahmin Edilen Sayı: ' + responseData.predicted_label);
        } else {
          console.error('An error occurred while uploading the file:', response);
          setResult('An error occurred while uploading the file.');
        }
      } catch (error) {
        console.error('An error occurred while uploading the file:', error);
        //setResult('An error occurred while uploading the file: ' + error.message);
      }
    } catch (error) {
      console.error('An error occurred while uploading the file:', error);
      setResult('An error occurred while uploading the file: ' + error.message);
    }
  };

  return (
    
    <div className="container">
      <div className="upload-section">
        <input class="form-control" type="file" accept="image/*" onChange={handleFileChange} />
        <div className="image-preview">
          {selectedFile && <img src={URL.createObjectURL(selectedFile)} alt="Preview" />}
        </div>
      </div>
      <button className="button" onClick={handleUpload}>
        Upload
      </button>
      <div className="result">{result}</div>
    </div>
  );
};

export default App;

