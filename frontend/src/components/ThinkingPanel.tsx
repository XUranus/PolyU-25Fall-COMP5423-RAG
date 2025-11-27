import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ShimmerEffect.css'; // import the CSS above


export interface MessageHttpResponse {
  id: string;
  sender: string,
  content: string;
  thinking_process?: string;
  timestamp: string;
}

export interface RetrievedDoc {
  id: string;
  text: string;
  score: number;
}

export interface ThinkingProcessStep {
  step: number,
  description: string,
  type : string,
  result? : string,
  retrieved_docs? : RetrievedDoc[]
}

export interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  thinkingProcess?: ThinkingProcessStep[]; // Optional for bot messages
  timestamp: string;
}

export function convertHttpResponseToMessage(httpResponse: MessageHttpResponse): Message {
  // Basic mapping
  const convertedMessage: Message = {
    id: httpResponse.id,
    sender: httpResponse.sender as 'user' | 'bot', // Type assertion after confirming source
    content: httpResponse.content,
    timestamp: httpResponse.timestamp,
  };

  // Parse thinking_process if it exists
  if (httpResponse.thinking_process) {
    try {
      // Attempt to parse the JSON string into an array of strings
      const parsedThinkingProcess: unknown = JSON.parse(httpResponse.thinking_process);
      // Type guard to ensure it's an array of strings
      if (Array.isArray(parsedThinkingProcess)) {
        
        convertedMessage.thinkingProcess = parsedThinkingProcess as ThinkingProcessStep[];
      } else {
        console.warn(`Parsed thinking_process is not an array of strings for message ${httpResponse.id}. Got:`, parsedThinkingProcess);
        // Optionally, you could set it to an empty array or skip the field if parsing fails strictly
        // convertedMessage.thinkingProcess = [];
      }
    } catch (error) {
      console.error(`Failed to parse thinking_process JSON for message ${httpResponse.id}:`, error);
      console.error(`Raw thinking_process string was:`, httpResponse.thinking_process);
      // Optionally, add an error message to the thinkingProcess field
      // convertedMessage.thinkingProcess = [`Error parsing thinking process: ${error.message}`];
    }
  }
  return convertedMessage;
}


export const MarkdownRenderer: React.FC<{ markdownContent: string }> = ({ markdownContent }) => {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {markdownContent}
      </ReactMarkdown>
    </div>
  );
};

// ThinkingGlow
interface ThinkingGlowProps {
  isThinking: boolean; // Receive the active chat ID from App
  text : string
}

export const ThinkingGlow : React.FC<ThinkingGlowProps> = ({ isThinking , text }) => {
  if (!isThinking) return null;

  return (
    <div className="shimmereffect-text">
      {text}
    </div>
  );
};

// CommonThinkingStep
interface CommonThinkingStepProps {
  description : string
}

const CommonThinkingStep : React.FC<CommonThinkingStepProps> = ({description}) => {
  return (
    <div className="flex items-start">
      <span className="mr-1">•&nbsp;</span>
      <div className="flex-1">
        <strong><MarkdownRenderer markdownContent={description} /></strong>
      </div>
    </div>
  )
}


// MultiHopSubQuestionGeneration
interface MultiHopSubQuestionGenerationProps {
  description : string,
  answer : string
}

const MultiHopSubQuestionGeneration : React.FC<MultiHopSubQuestionGenerationProps> = ({description, answer}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex items-start">
      <span className="mr-1">•</span>
      <div className="flex-1">
        <div 
          className={`cursor-pointer hover:bg-gray-600 p-1 rounded ${isOpen ? 'bg-gray-600' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <strong><MarkdownRenderer markdownContent={description} /></strong>
        </div>
        {isOpen && (
          <div className="ml-4 mt-1 space-y-1">
            <MarkdownRenderer markdownContent={answer} />
          </div>
        )}
      </div>
    </div>
  )
}

// QueryWithDocuments
interface QueryWithDocumentsProps {
  description : string,
  documents : RetrievedDoc[]
}

const QueryWithDocuments : React.FC<QueryWithDocumentsProps> = ({description, documents}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex items-start">
      <span className="mr-1">•</span>
      <div className="flex-1">
        <div 
          className={`cursor-pointer hover:bg-gray-600 p-1 rounded ${isOpen ? 'bg-gray-600' : ''}`}
          onClick={() => setIsOpen(!isOpen)}
        >
          <strong><MarkdownRenderer markdownContent={description} /></strong>
        </div>
        {isOpen && (
          <div className="ml-4 mt-1 space-y-2">
            {documents.map((doc, index) => (
              <div key={index} className="p-2 bg-gray-800 rounded">
                <div className="text-xs text-gray-400">
                  <span className="font-medium">Doc ID:</span> {doc.id} | 
                  <span className="font-medium"> Score:</span> {doc.score.toFixed(4)}
                </div>
                <div className="mt-1">
                  <MarkdownRenderer markdownContent={doc.text} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}


// ThinkingStepDiv
interface ThinkingStepDivProps {
  step : ThinkingProcessStep
}

const ThinkingStepDiv : React.FC<ThinkingStepDivProps> = ({step}) => {
  if (step.type === 'multi_hop_sub_generation') {
    return (
      <MultiHopSubQuestionGeneration
        answer={step.result || ''} 
        description={step.description}/>
    )
  } else if (step.type === 'multi_hop_sub_retrieval' || step.type === 'single_hop_retrieval') {
    return (
      <QueryWithDocuments 
        description={step.description}
        documents={step.retrieved_docs || []}/>
    )
  } else {
    return (
      <CommonThinkingStep
        description={step.description}/>
    )
  } 
}


// ThinkingPanel
interface ThinkingPanelProps {
  steps : ThinkingProcessStep[]
}

const ThinkingPanel : React.FC<ThinkingPanelProps> = ({steps}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mb-2 p-2 bg-gray-700 rounded text-sm">
      <div 
        className={`cursor-pointer hover:bg-gray-600 p-1 rounded ${isOpen ? 'bg-gray-600' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <strong>💡 View Thinking Process ({steps.length} steps)</strong>
      </div>
      {isOpen && (
        <div className="mt-2 space-y-2">
          {steps.map((step) => (
            <ThinkingStepDiv step={step} key={step.step}/>
          ))}
        </div>
      )}
    </div>
  )
}

export default ThinkingPanel;