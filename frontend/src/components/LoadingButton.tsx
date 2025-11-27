import React from 'react';

interface LoadingButtonProps {
  loading : boolean,
  children : any,
  noinput : boolean,
  disabled : boolean,
  onClick : () => void
}

const LoadingButton : React.FC<LoadingButtonProps> = ({ loading, children, noinput, disabled, onClick }) => {
  const handleClick = () => {
    // Only call onClick if the button is not disabled
    if (!disabled) {
      onClick();
    }
  };

  return (
    <button
      disabled={disabled} // Use the disabled prop instead of just loading
      onClick={handleClick}
      className={`flex items-center justify-center px-4 py-2 rounded-md font-medium
        ${disabled ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}
        text-white transition-colors`}
    >
      {loading ? (
        <>
          <svg
            className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.707z"
            ></path>
          </svg>
          Loading...
        </>
      ) : (
        children
      )}
    </button>
  );
};

export default LoadingButton;