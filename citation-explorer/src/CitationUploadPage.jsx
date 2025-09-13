import React, { useState } from "react";
import { Upload, Link as LinkIcon } from "lucide-react";

export default function CitationUploadPage({ onSubmitDOI }) {
  const [doi, setDoi] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    const clean = doi.trim();
    if (!clean) return;
    await onSubmitDOI(clean);
  };

  return (
    <div className="w-full h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-white p-6">
      <div className="max-w-xl w-full bg-white shadow-xl rounded-3xl p-10 flex flex-col items-center gap-8">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          Watership Down 🐇
        </h1>
        <form onSubmit={handleSubmit} className="w-full flex flex-col gap-6">
          <div className="flex items-center gap-3 border-b border-slate-300 focus-within:border-slate-500">
            <LinkIcon className="w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={doi}
              onChange={(e) => setDoi(e.target.value)}
              placeholder="Enter DOI (e.g., 10.1038/s41586-020-03167-3)"
              className="flex-1 py-2 px-1 bg-transparent focus:outline-none text-slate-700"
            />
          </div>
          <button
            type="submit"
            className="mt-4 w-full py-3 rounded-2xl bg-slate-900 text-white font-medium tracking-wide hover:bg-slate-800 transition"
          >
            Continue
          </button>
        </form>
      </div>
    </div>
  );
}
