import React, { useEffect, useState } from 'react';
import './App.css';
import { FaGithub } from "react-icons/fa";
import { RingLoader } from 'react-spinners';

const App = () => {
  const metricsUrl = 'http://127.0.0.1:5000/metrics';
  const predictUrl = 'http://127.0.0.1:5000/predict';

  const [url, setUrl] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [isMetricsLoading, setIsMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState('');
  const [predictionError, setPredictionError] = useState('');
  const [predictionModel, setPredictionModel] = useState('');

  useEffect(() => {
    setIsMetricsLoading(true);
    fetch(metricsUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Unable to load model accuracy metrics');
        }
        return response.json();
      })
      .then((data) => {
        setMetrics(data);
        setMetricsError('');
      })
      .catch((error) => {
        console.error('Error:', error);
        setMetricsError('Could not load accuracy metrics');
      })
      .finally(() => {
        setIsMetricsLoading(false);
      });
  }, [metricsUrl]);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!url.trim()) {
      setPredictionError('Please enter a URL to check.');
      setResult('');
      return;
    }

    setIsLoading(true);
    setPredictionError('');
    setResult('');
    setPredictionModel('');
    const payload = { url };
    fetch(predictUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error('Prediction request failed');
        }
        return response.json();
      })
      .then(data => {
        setIsLoading(false);
        setResult(data.data);
        setPredictionModel(data.model || '');
      })
      .catch(error => {
        setIsLoading(false);
        console.error('Error:', error);
        setPredictionError('Could not process this URL right now.');
    });
  };

  const formatAccuracy = (value) => {
    if (typeof value !== 'number') {
      return '--';
    }
    return `${value.toFixed(2)}%`;
  };

  const formatTimestamp = (value) => {
    if (!value) {
      return '--';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return '--';
    }
    return date.toLocaleString();
  };

  const hasResult = result.trim().length > 0;
  const isPhishyResult = result.toLowerCase().includes('phishy');

  return (
    <div className='dashboardShell py-12 px-6'>
      <a href="https://www.github.com/" className='right-5 top-5 z-10 absolute' >
          <FaGithub size={30} color={"white"}/>
          </a>

      <div className="content">
        <h1 className="title">Phishing Detection Dashboard
          <div className="aurora">
            <div className="aurora__item"></div>
            <div className="aurora__item"></div>
            <div className="aurora__item"></div>
            <div className="aurora__item"></div>
          </div>
        </h1>
        <p className='dashboardSubtitle'>
          Live predictor with baseline, attack, and defence performance.
        </p>
      </div>

      <div className='metricsPanel mt-10 mx-auto md:w-[75%] w-full'>
        <h2 className='metricsTitle'>Model Performance</h2>
        {isMetricsLoading ? (
          <p className='metricsLoading'>Loading model accuracy...</p>
        ) : (
          <div className='metricsGrid'>
            <div className='metricCard'>
              <p className='metricLabel'>Base Model</p>
              <p className='metricValue'>{formatAccuracy(metrics?.base_model_accuracy)}</p>
            </div>
            <div className='metricCard'>
              <p className='metricLabel'>After Attack</p>
              <p className='metricValue'>{formatAccuracy(metrics?.after_attack_accuracy)}</p>
            </div>
            <div className='metricCard'>
              <p className='metricLabel'>Defence</p>
              <p className='metricValue'>{formatAccuracy(metrics?.defence_accuracy)}</p>
            </div>
            <div className='metricCard'>
              <p className='metricLabel'>Robustness Gain</p>
              <p className='metricValue'>{formatAccuracy(metrics?.robustness_gain_pp)}</p>
            </div>
          </div>
        )}
        {metricsError && <p className='metricsError'>{metricsError}</p>}
        {!isMetricsLoading && !metricsError && (
          <div className='dashboardMeta'>
            <p><span>Model:</span> {metrics?.model_name || '--'}</p>
            <p><span>Defence:</span> {metrics?.defense_name || '--'}</p>
            <p><span>Metrics Updated:</span> {formatTimestamp(metrics?.metrics_updated_at)}</p>
          </div>
        )}
      </div>

      <div className='predictPanel mt-8 mx-auto md:w-[75%] w-full'>
        <h2 className='predictTitle'>Live URL Check</h2>
        <form id="fileForm" className='mx-auto w-full' onSubmit={handleSubmit}>
            <label className="input input-bordered rounded-xl max-md:text-sm flex items-center gap-3 pr-0 py-1">
              URL :  
            <input type="text"
              className="grow border-l bg-transparent max-md:text-sm pl-4" 
              placeholder="Enter URL (example: https://example.com)" 
              id="urlInput"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button className="btn btn-active btn-primary btn-sm mr-2"  type="submit">{isLoading? <RingLoader  color={"#ffffff"} size={26}/> : "Check URL"}</button>
          </label>
        </form>
        {predictionError && <p className='metricsError mt-4'>{predictionError}</p>}

        <div className={`resultPanel ${result ? 'resultPanelActive' : ''}`}>
          <p className='resultHeading'>Prediction</p>
          <p
            id="result"
            className={`poppins-medium resultText ${
              hasResult
                ? (isPhishyResult ? "text-[#FF3131] redShadow" : "text-neonGreen greenShadow")
                : "resultPlaceholder"
            }`}
          >
            {result || 'Submit a URL to view the prediction.'}
          </p>
          {predictionModel && <p className='resultMeta'>Model used: {predictionModel}</p>}
        </div>
      </div>

    </div>
  );
};

export default App;
