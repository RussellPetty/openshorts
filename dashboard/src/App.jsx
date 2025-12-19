import React, { useState, useEffect } from 'react';
import { Sparkles, Activity } from 'lucide-react';
import KeyInput from './components/KeyInput';
import MediaInput from './components/MediaInput';
import ResultCard from './components/ResultCard';

// Mock polling function
const pollJob = async (jobId) => {
  const res = await fetch(`/api/status/${jobId}`);
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
};

function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_key') || '');
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [results, setResults] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logsVisible, setLogsVisible] = useState(false);

  useEffect(() => {
    if (apiKey) localStorage.setItem('gemini_key', apiKey);
  }, [apiKey]);

  useEffect(() => {
    let interval;
    if (status === 'processing' && jobId) {
      interval = setInterval(async () => {
        try {
          const data = await pollJob(jobId);
          console.log("Job status:", data);
          if (data.status === 'completed') {
            setResults(data.result);
            setStatus('complete');
            clearInterval(interval);
          } else if (data.status === 'failed') {
            setStatus('error');
            setLogs(prev => [...prev, "Error: " + data.error]);
            clearInterval(interval);
          } else {
            // Update logs if available
            if (data.logs) setLogs(data.logs);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [status, jobId]);

  const handleProcess = async (data) => {
    setStatus('processing');
    setLogs(["Starting process..."]);
    setResults(null);

    try {
      let body;
      const headers = { 'X-Gemini-Key': apiKey };

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({ url: data.payload });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        body = formData;
        // Content-Type is auto-set for FormData
      }

      const res = await fetch('/api/process', {
        method: 'POST',
        headers: data.type === 'url' ? headers : { 'X-Gemini-Key': apiKey },
        body
      });

      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();
      setJobId(resData.job_id);

    } catch (e) {
      setStatus('error');
      setLogs(l => [...l, `Error starting job: ${e.message}`]);
    }
  };

  return (
    <div className="min-h-screen bg-background relative overflow-x-hidden p-6 md:p-12">
      {/* Background Gradients */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] bg-primary/10 rounded-full blur-[120px]" />
        <div className="absolute top-[20%] -right-[10%] w-[40%] h-[60%] bg-accent/10 rounded-full blur-[120px]" />
      </div>

      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center animate-[fadeIn_0.5s_ease-out]">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/10 mb-4">
            <Sparkles size={14} className="text-yellow-400" />
            <span className="text-xs font-medium text-zinc-300">AI Viral Clipper</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black bg-gradient-to-br from-white via-white to-white/50 bg-clip-text text-transparent mb-4 tracking-tight">
            OpenShorts<span className="text-primary">.app</span>
          </h1>
          <p className="text-zinc-400 text-lg max-w-lg mx-auto">
            Transform long-form videos into viral Shorts, Reels, and TikToks instantly with Gemini AI.
          </p>
        </header>

        <KeyInput onKeySet={setApiKey} savedKey={apiKey} />

        <div className="space-y-8">
          <MediaInput onProcess={handleProcess} isProcessing={status === 'processing'} />

          {(status === 'processing' || status === 'error') && (
            <div className={`glass-panel p-6 ${status === 'error' ? 'border-red-500/30' : ''}`}>
              <div className="flex items-center justify-between mb-4">
                <div className={`flex items-center gap-3 ${status === 'error' ? 'text-red-400' : 'text-zinc-300'}`}>
                  <Activity size={20} className={status === 'processing' ? "animate-pulse text-primary" : ""} />
                  <span className="font-mono text-sm">{status === 'error' ? 'Process Failed' : 'Processing Status'}</span>
                </div>
                <button
                  onClick={() => setLogsVisible(!logsVisible)}
                  className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  {logsVisible ? 'Hide Logs' : 'Show Logs'}
                </button>
              </div>

              {logsVisible && (
                <div className="h-48 overflow-y-auto font-mono text-xs text-zinc-400 space-y-1 custom-scrollbar bg-black/20 p-4 rounded-lg animate-[fadeIn_0.2s_ease-out]">
                  {logs.map((log, i) => (
                    <div key={i} className={log.toLowerCase().includes('error') || log.toLowerCase().includes('failed') ? 'text-red-400' : ''}>
                      {'>'} {log}
                    </div>
                  ))}
                  {status === 'processing' && <div className="animate-pulse">{'>'} _</div>}
                </div>
              )}
              {!logsVisible && status === 'processing' && (
                <div className="text-zinc-500 text-sm animate-pulse">
                  Analyzing video content with Gemini AI...
                </div>
              )}
            </div>
          )}

          {status === 'error' && (
            <div className="mt-4 bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-center text-sm">
              Processing encountered an error.
              <button
                onClick={() => setLogsVisible(true)}
                className="ml-2 underline hover:text-white"
              >
                Check Logs
              </button>
              <button
                onClick={() => setStatus('idle')}
                className="ml-4 underline hover:text-white"
              >
                Try Again
              </button>
            </div>
          )}

          {status === 'complete' && results && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Sparkles className="text-yellow-400" /> Results Generated
              </h2>
              <div className="grid gap-6">
                {results.clips.map((clip, i) => (
                  <ResultCard key={i} clip={clip} index={i} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
