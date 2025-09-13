import React, { useState } from "react";
import { Upload, Link as LinkIcon } from "lucide-react";

export default function CitationUploadPage() {
  const [doi, setDoi] = useState("");
  const [fileName, setFileName] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setFileName(file.name);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Submitting:", { doi, fileName });
  };

  return (
    <div className="w-full h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-white p-6">
      <div className="max-w-xl w-full bg-white shadow-xl rounded-3xl p-10 flex flex-col items-center gap-8">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          Start Your Citation Journey
        </h1>

        <form onSubmit={handleSubmit} className="w-full flex flex-col gap-6">
          {/* DOI Input */}
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

          {/* File Upload */}
          <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 hover:border-slate-400 rounded-2xl py-10 cursor-pointer transition">
            <Upload className="w-6 h-6 text-slate-400 mb-2" />
            <span className="text-sm text-slate-500">
              {fileName ? fileName : "Upload PDF of paper"}
            </span>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

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
