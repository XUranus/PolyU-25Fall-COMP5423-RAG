import React from 'react';
import { useEffect, useState } from 'react';
import { API_PREFIX } from './config'


export interface BackendStatus {
  ok : boolean,
  storage : string,
  cache : string,
  logger : string,
  db : string,
  ready : boolean
}

interface InitPageProps {
}

const InitPage: React.FC<InitPageProps> = ({}) => {
  const [backendStatus, SetBackendStatus] = useState<BackendStatus>({ ok : false, storage : "", cache : "", logger : "", db : "", ready : false });
  
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(API_PREFIX + '/api/health');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: BackendStatus = await response.json();
        if (data.ok && data.ready) {
          window.location.reload(); // or navigate to main app
        } else if (data.ok) { // server connected but not inited
          SetBackendStatus(data)
        }
      } catch (err) {
        // Server not up yet — silently ignore
      }
    };

    const intervalId = setInterval(checkHealth, 5000); // every 5s

    // Cleanup on unmount (optional but good practice)
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex flex-col items-center justify-center p-4">
      {/* Logo Placeholder - Replace with your actual logo */}
      <div className="mb-8">
        <div className="bg-indigo-600 text-white font-bold text-4xl w-40 h-40 rounded-xl flex items-center justify-center">
          RAG42
        </div>
      </div>
      
      {/* Spinning Circle */}
      <div className="relative w-16 h-16 mb-6">
        <div className="absolute inset-0 border-4 border-t-indigo-500 border-r-indigo-500 border-b-transparent border-l-transparent rounded-full animate-spin"></div>
      </div>
      
      {/* Text */}
      <h1 className="text-2xl font-semibold text-white text-center">
        {backendStatus.ok ? "Server Connected, Waiting For RAG Module To Be Initialized..." : "Connecting To Server..."}
      </h1>
      <p className="text-gray-400 mt-2 text-center max-w-md">Backend API prefix : {API_PREFIX}</p>
      {backendStatus.ok ? (<p className="text-gray-400 mt-2 text-center max-w-md">Backend Logger : {backendStatus.logger}</p>) : null}
      {backendStatus.ok ? (<p className="text-gray-400 mt-2 text-center max-w-md">Backend Database : {backendStatus.db}</p>) : null}
      {backendStatus.ok ? (<p className="text-gray-400 mt-2 text-center max-w-md">Backend Cache : {backendStatus.cache}</p>) : null}
      {backendStatus.ok ? (<p className="text-gray-400 mt-2 text-center max-w-md">Backend Storage : {backendStatus.storage}</p>) : null}
      <p className="text-gray-400 mt-2 text-center max-w-md">
        GitHub: <a href='https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG'>PolyU-25Fall-COMP5423-RAG</a>
      </p>
    </div>
  );
};

export default InitPage;