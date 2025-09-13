import { useState, useEffect } from "react";
import CitationUploadPage, { CitationRightPane } from "./CitationUploadPage";

export default function App() {
  const [view, setView] = useState("upload");
  const [data, setData] = useState(null);

  // load dummy dataset once
  useEffect(() => {
    fetch("/dummy-dataset.json")
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error("Failed to load dataset", err));
  }, []);

  return (
    <div className="h-screen">
      <div className="p-3 flex gap-2">
        <button
          onClick={() => setView("upload")}
          className={`px-3 py-1 rounded-xl border ${
            view === "upload"
              ? "bg-slate-900 text-white"
              : "bg-white"
          }`}
        >
          Upload
        </button>
        <button
          onClick={() => setView("viz")}
          className={`px-3 py-1 rounded-xl border ${
            view === "viz"
              ? "bg-slate-900 text-white"
              : "bg-white"
          }`}
        >
          Visualization
        </button>
      </div>

      {view === "upload" ? (
        <CitationUploadPage />
      ) : data ? (
        <CitationRightPane data={data} />
      ) : (
        <div className="p-6 text-slate-500">Loading dataset…</div>
      )}
    </div>
  );
}
