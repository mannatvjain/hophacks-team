import { useState } from "react";
import CitationUploadPage from "./CitationUploadPage";
import { CitationRightPane } from "./CitationUploadPage"; // the right-pane export in your file

export default function App() {
  const [view, setView] = useState("upload"); // "upload" | "viz"

  return (
    <div className="h-screen">
      <div className="p-3 flex gap-2">
        <button
          onClick={() => setView("upload")}
          className={`px-3 py-1 rounded-xl border ${view==='upload' ? 'bg-slate-900 text-white' : 'bg-white'}`}
        >
          Upload
        </button>
        <button
          onClick={() => setView("viz")}
          className={`px-3 py-1 rounded-xl border ${view==='viz' ? 'bg-slate-900 text-white' : 'bg-white'}`}
        >
          Visualization
        </button>
      </div>

      {view === "upload" ? <CitationUploadPage /> : <CitationRightPane />}
    </div>
  );
}


