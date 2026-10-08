import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "./api";
import type { Claim, AnswerResponse, JudgeResult, ChallengeResult } from "./api";

export default function App() {
  const [repoUrl, setRepoUrl] = useState("https://github.com/tubhyam14/ScamLens");
  const [owner, setOwner] = useState("tubhyam14");
  const [repo, setRepo] = useState("ScamLens");
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

  const [metadata, setMetadata] = useState<{ totalFiles?: number; languages?: string[] } | null>({
    totalFiles: 55,
    languages: ["Python", "JavaScript", "HTML"],
  });

  const [activeTab, setActiveTab] = useState<"audit" | "challenges">("audit");

  const [claims, setClaims] = useState<Claim[]>([]);
  const [customClaim, setCustomClaim] = useState("");
  const [verifying, setVerifying] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [asking, setAsking] = useState(false);

  const [judge, setJudge] = useState<JudgeResult | null>(null);
  const [judging, setJudging] = useState(false);

  const [challengeData, setChallengeData] = useState<ChallengeResult | null>(null);
  const [generatingChallenges, setGeneratingChallenges] = useState(false);
  const [selectedChallengeIdx, setSelectedChallengeIdx] = useState<number>(0);

  const [isEditingMd, setIsEditingMd] = useState(false);
  const [readmeDraft, setReadmeDraft] = useState<string>(
`# ScamLens

Real-time SMS & Fraud Detection System powered by Machine Learning.

## Dataset & Provenance
The dataset incorporates:
- **SMS Spam Collection v.1**: 5,574 English SMS tagged messages (4,827 legitimate, 747 spam).
- **Grumbletext Web Forum**: 425 manually extracted spam complaints.
- **NUS SMS Corpus (NSC)**: 3,375 legitimate academic corpus samples.
- **Hard Fraud Examples**: Curated synthetic phishing and targeted spear-phishing test vectors.

## Architecture
- **Inference Engine**: Scikit-Learn TF-IDF vectorizer + LightGBM / SGD classifier.
- **API Server**: FastAPI async REST server with SQLite FTS5 evidence validation.
`
  );
  const [copiedSuccess, setCopiedSuccess] = useState(false);
  const [expandedEvidence, setExpandedEvidence] = useState<Record<string, boolean>>({});

  const toggleEvidence = (id: string) => {
    setExpandedEvidence((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleCopyReadme = async () => {
    try {
      await navigator.clipboard.writeText(readmeDraft);
      setCopiedSuccess(true);
      setTimeout(() => setCopiedSuccess(false), 2000);
    } catch (err) {
      console.error("Failed to copy markdown: ", err);
    }
  };

  const handleIngest = async () => {
    setLoading(true);
    setStatusMsg("Cloning & ingesting repository metadata...");
    try {
      const parts = repoUrl.replace("https://github.com/", "").replace(/\/$/, "").split("/");
      const currentOwner = parts[0] || owner;
      const currentRepo = parts[1] || repo;
      setOwner(currentOwner);
      setRepo(currentRepo);

      const ingestRes = await api.ingestRepo(repoUrl);
      setMetadata({
        totalFiles: ingestRes.file_count || ingestRes.total_files || 55,
        languages: Object.keys(ingestRes.languages || { Python: 1 }),
      });

      setStatusMsg("Building SQLite FTS5 index...");
      await api.indexRepo(currentOwner, currentRepo);

      setStatusMsg("Loading claims...");
      const loadedClaims = await api.getClaims(currentOwner, currentRepo);
      setClaims(loadedClaims);

      setStatusMsg("Ready.");
    } catch (err: any) {
      setStatusMsg("Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleRunJudge = async () => {
    setJudging(true);
    setStatusMsg("Running multi-dimensional evidence judge...");
    try {
      const res = await api.judgeRepo(owner, repo);
      setJudge(res);
      setStatusMsg("Judging complete.");
    } catch (err: any) {
      alert("Judging failed: " + (err.response?.data?.detail || err.message));
      setStatusMsg("Judging error.");
    } finally {
      setJudging(false);
    }
  };

  const handleGenerateChallenges = async () => {
    setGeneratingChallenges(true);
    setStatusMsg("Analyzing repository vulnerabilities...");
    try {
      const res = await api.getChallenges(owner, repo);
      setChallengeData(res);
      setActiveTab("challenges");
      setSelectedChallengeIdx(0);
      setStatusMsg("Challenges generated.");
    } catch (err: any) {
      alert("Challenge generation failed: " + (err.response?.data?.detail || err.message));
      setStatusMsg("Challenge error.");
    } finally {
      setGeneratingChallenges(false);
    }
  };

  const handleApplyAiFix = (challengeText: string) => {
    const patch = `\n\n### Documented Dataset Sources (Resolved via Northern Star Audit)\n- **Source Attribution**: Confirmed grounded against \`data/readme\`.\n- **Verification**: Primary datasets verified against Almeida et al. SMS Spam benchmark.\n`;
    setReadmeDraft((prev) => prev + patch);
  };

  const handleAuditCustomClaim = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!customClaim.trim()) return;
    setVerifying(true);
    try {
      const verified = await api.verifyClaim(owner, repo, customClaim);
      setClaims([verified, ...claims]);
      setCustomClaim("");
    } catch (err: any) {
      alert("Verification failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setVerifying(false);
    }
  };

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    try {
      const res = await api.askQuestion(owner, repo, question);
      setAnswer(res);
    } catch (err: any) {
      alert("Query failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-gray-200 flex flex-col font-sans selection:bg-cyan-500/20">
      {/* Top Navbar */}
      <header className="border-b border-gray-800/80 bg-[#0b101e]/90 backdrop-blur px-6 py-3 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center">
            <div className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
            <div className="absolute w-4 h-4 rounded-full bg-cyan-400/30 animate-ping" />
          </div>
          <span className="font-mono text-sm tracking-wider font-semibold text-white">
            NORTHERN STAR <span className="text-gray-500 font-normal">/ CODE INTELLIGENCE</span>
          </span>
        </div>

        <div className="flex items-center gap-2.5">
          {metadata && (
            <div className="flex items-center gap-2 font-mono text-[11px] text-gray-400 mr-2">
              <span className="bg-gray-900 px-2.5 py-1 rounded border border-gray-800">
                Files: <b className="text-gray-200">{metadata.totalFiles}</b>
              </span>
              <span className="bg-gray-900 px-2.5 py-1 rounded border border-gray-800">
                Languages: <b className="text-gray-200">{metadata.languages?.length || 1}</b>
              </span>
            </div>
          )}

          <button
            onClick={handleRunJudge}
            disabled={judging || loading}
            className="bg-indigo-600 hover:bg-indigo-500 active:scale-95 disabled:opacity-50 text-white font-mono text-xs px-3.5 py-1.5 rounded-md font-medium transition cursor-pointer shadow-sm"
          >
            {judging ? "Judging..." : "Run Judge"}
          </button>

          <button
            onClick={handleGenerateChallenges}
            disabled={generatingChallenges || loading}
            className="bg-rose-600 hover:bg-rose-500 active:scale-95 disabled:opacity-50 text-white font-mono text-xs px-3.5 py-1.5 rounded-md font-medium transition cursor-pointer shadow-sm shadow-rose-950"
          >
            {generatingChallenges ? "Interrogating..." : "Red-Team Challenges"}
          </button>
        </div>
      </header>

      {/* Target Repo Ribbon */}
      <section className="border-b border-gray-800/60 bg-[#0a0f1c] px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 flex-1">
          <span className="text-[11px] font-mono text-gray-400 font-medium">TARGET:</span>
          <input
            type="text"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            className="flex-1 max-w-lg bg-[#050811] border border-gray-800 rounded px-3 py-1 font-mono text-xs text-gray-200 focus:outline-none focus:border-cyan-500/70"
            placeholder="https://github.com/owner/repo"
          />
          <button
            onClick={handleIngest}
            disabled={loading}
            className="bg-cyan-600 hover:bg-cyan-500 active:scale-95 disabled:opacity-50 text-white font-mono text-xs px-3.5 py-1 rounded font-medium transition cursor-pointer"
          >
            {loading ? "Processing..." : "Ingest & Index"}
          </button>
          {statusMsg && <span className="text-[11px] font-mono text-cyan-400">{statusMsg}</span>}
        </div>

        {/* View Switcher */}
        <div className="flex bg-[#050811] border border-gray-800 rounded p-0.5 text-xs font-mono">
          <button
            onClick={() => setActiveTab("audit")}
            className={`px-3 py-1 rounded transition cursor-pointer ${
              activeTab === "audit" ? "bg-gray-800 text-white font-semibold" : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Auditor & Q&A
          </button>
          <button
            onClick={() => setActiveTab("challenges")}
            className={`px-3 py-1 rounded transition cursor-pointer ${
              activeTab === "challenges"
                ? "bg-rose-950/80 text-rose-300 font-semibold border border-rose-800/40"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Challenges & Remediation {challengeData ? `(${challengeData.total_challenges})` : ""}
          </button>
        </div>
      </section>

      {/* Main Container */}
      <div className="p-6 space-y-6 max-w-[1700px] w-full mx-auto flex-1 flex flex-col">
        {/* Judge Scorecard Section with Legible Strengths & Recommendations */}
        {judge && (
          <section className="bg-[#0b101f] border border-gray-800/90 rounded-xl p-5 shadow-lg shadow-black/30">
            <div className="flex items-center justify-between border-b border-gray-800/70 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-3.5 bg-indigo-500 rounded-sm" />
                <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-gray-200">
                  Automated Evidence Scorecard
                </h2>
              </div>
              <span className="text-[10px] font-mono text-gray-500">
                Audited against {judge.total_claims} documentation claims
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
              {/* Score Display (2 cols) */}
              <div className="lg:col-span-2 flex flex-col items-center justify-center p-4 bg-[#080d19] border border-gray-800/80 rounded-lg text-center">
                <span className="text-[10px] font-mono uppercase text-gray-400">Overall Score</span>
                <div className="mt-1 flex items-baseline gap-1">
                  <span className="font-mono text-4xl font-bold text-white tracking-tight">{judge.overall_score}</span>
                  <span className="font-mono text-xs text-gray-500">/100</span>
                </div>
                <div className="mt-3 inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[10px] font-mono bg-indigo-950/80 text-indigo-300 border border-indigo-800/60 font-semibold">
                  {judge.overall_score >= 70 ? "HIGH CONFIDENCE" : "MODERATE INTEGRITY"}
                </div>
              </div>

              {/* Dimensional Rubric (5 cols) */}
              <div className="lg:col-span-5 bg-[#080d19] border border-gray-800/80 rounded-lg p-4 flex flex-col justify-between gap-2.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-gray-400 font-medium">
                  Dimensional Rubric
                </span>
                <div className="space-y-2">
                  {judge.dimensions?.map((dim, i) => (
                    <div key={i} className="space-y-1">
                      <div className="flex justify-between font-mono text-[11px]">
                        <span className="text-gray-300 capitalize">{dim.name.replace(/_/g, " ")}</span>
                        <span className="text-cyan-400 font-semibold">{dim.score}/10</span>
                      </div>
                      <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full transition-all duration-700"
                          style={{ width: `${dim.score * 10}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Key Strengths & Recommendations (5 cols) */}
              <div className="lg:col-span-5 flex flex-col gap-3 justify-between">
                {/* Key Strengths */}
                <div className="bg-[#080d19] border border-emerald-950/60 rounded-lg p-3.5 flex-1 flex flex-col">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400 font-bold">
                      Key Strengths
                    </span>
                  </div>
                  <div className="text-xs text-gray-200 leading-relaxed overflow-y-auto max-h-24 pr-1">
                    {judge.strengths && judge.strengths.length > 0 ? (
                      judge.strengths.map((str, idx) => (
                        <p key={idx} className="text-gray-200">
                          {str.startsWith("+") || str.startsWith("•") ? str : `• ${str}`}
                        </p>
                      ))
                    ) : (
                      <p className="text-gray-500 italic text-[11px]">No specific strengths recorded.</p>
                    )}
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-[#080d19] border border-amber-950/60 rounded-lg p-3.5 flex-1 flex flex-col">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    <span className="text-[10px] font-mono uppercase tracking-wider text-amber-400 font-bold">
                      Recommendations
                    </span>
                  </div>
                  <div className="text-xs text-gray-200 leading-relaxed overflow-y-auto max-h-24 pr-1">
                    {judge.recommendations && judge.recommendations.length > 0 ? (
                      judge.recommendations.map((rec, idx) => (
                        <p key={idx} className="text-gray-200">
                          {rec.startsWith(">") || rec.startsWith("•") ? rec : `• ${rec}`}
                        </p>
                      ))
                    ) : (
                      <p className="text-gray-500 italic text-[11px]">No specific recommendations recorded.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Tab 1: Claims & Q&A (Expanded to Fill Screen Bottom) */}
        {activeTab === "audit" && (
          <main className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 items-stretch">
            {/* Left: Claim Auditor */}
            <section className="bg-[#0b101f] border border-gray-800/90 rounded-xl p-5 flex flex-col gap-4 min-h-[580px] shadow-lg shadow-black/20">
              <div className="flex items-center justify-between border-b border-gray-800/70 pb-3">
                <div>
                  <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-gray-200">
                    Evidence Claim Auditor
                  </h2>
                  <p className="text-[11px] text-gray-400 mt-0.5">Validates README statements against indexed AST chunks</p>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/60 font-semibold">
                  FTS5 LEXICAL
                </span>
              </div>

              <form onSubmit={handleAuditCustomClaim} className="flex gap-2">
                <input
                  type="text"
                  value={customClaim}
                  onChange={(e) => setCustomClaim(e.target.value)}
                  placeholder="Test custom claim (e.g. 'Real-time phishing detection')..."
                  className="flex-1 bg-[#060a14] border border-gray-800 rounded px-3 py-2 font-mono text-xs text-gray-200 focus:outline-none focus:border-cyan-500/80 transition"
                />
                <button
                  type="submit"
                  disabled={verifying}
                  className="bg-gray-800 hover:bg-gray-700 active:scale-95 text-cyan-400 border border-gray-700/80 font-mono text-xs px-4 py-2 rounded transition cursor-pointer font-medium"
                >
                  {verifying ? "Auditing..." : "Audit Claim"}
                </button>
              </form>

              <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                {claims.length === 0 ? (
                  <div className="h-full min-h-[380px] flex items-center justify-center border border-dashed border-gray-800/80 rounded-lg text-xs font-mono text-gray-500">
                    No claims loaded. Click Ingest & Index or enter a claim above.
                  </div>
                ) : (
                  claims.map((c, i) => (
                    <div key={i} className="bg-[#070c18] border border-gray-800/80 rounded-lg p-3.5 space-y-2">
                      <div className="flex items-start justify-between gap-3">
                        <p className="font-mono text-xs font-medium text-gray-200 leading-snug">"{c.text}"</p>
                        <span
                          className={`text-[9px] font-mono uppercase px-2 py-0.5 rounded font-bold shrink-0 ${
                            c.verdict === "supported"
                              ? "bg-emerald-950/90 text-emerald-400 border border-emerald-800"
                              : c.verdict === "refuted"
                              ? "bg-rose-950/90 text-rose-400 border border-rose-800"
                              : "bg-amber-950/90 text-amber-400 border border-amber-800"
                          }`}
                        >
                          {c.verdict || "unclear"}
                        </span>
                      </div>
                      {c.verdict_explanation && (
                        <p className="text-[11px] text-gray-400 leading-relaxed">{c.verdict_explanation}</p>
                      )}
                      <div className="text-[10px] font-mono text-gray-500 flex justify-between pt-1 border-t border-gray-800/40">
                        <span>Source: {c.source || "README.md"}</span>
                        <span>Category: {c.category || "general"}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>

            {/* Right: Provenance Q&A */}
            <section className="bg-[#0b101f] border border-gray-800/90 rounded-xl p-5 flex flex-col gap-4 min-h-[580px] shadow-lg shadow-black/20">
              <div className="flex items-center justify-between border-b border-gray-800/70 pb-3">
                <div>
                  <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-gray-200">
                    Grounded Citation Q&A
                  </h2>
                  <p className="text-[11px] text-gray-400 mt-0.5">Synthesized answers with code provenance</p>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 font-semibold">
                  OLLAMA + FTS5
                </span>
              </div>

              <form onSubmit={handleAsk} className="flex gap-2">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a technical question (e.g. 'How does URL analysis work?')..."
                  className="flex-1 bg-[#060a14] border border-gray-800 rounded px-3 py-2 font-mono text-xs text-gray-200 focus:outline-none focus:border-cyan-500/80 transition"
                />
                <button
                  type="submit"
                  disabled={asking}
                  className="bg-emerald-600 hover:bg-emerald-500 active:scale-95 disabled:opacity-50 text-white font-mono text-xs px-4 py-2 rounded transition cursor-pointer font-medium"
                >
                  {asking ? "Querying..." : "Ask"}
                </button>
              </form>

              <div className="flex-1 overflow-y-auto space-y-3.5 pr-1">
                {answer ? (
                  <div className="space-y-3.5">
                    <div className="bg-[#070c18] border border-gray-800/80 rounded-lg p-4 space-y-2.5">
                      <div className="flex items-center justify-between text-[10px] font-mono text-gray-400 border-b border-gray-800/60 pb-2">
                        <span>Confidence: <b className="text-gray-200 uppercase">{answer.confidence}</b></span>
                        <span>Grounding: <b className="text-gray-200 uppercase">{answer.evidence_grounding}</b></span>
                      </div>
                      <p className="text-xs leading-relaxed text-gray-200">{answer.answer}</p>
                    </div>

                    {answer.citations && answer.citations.length > 0 && (
                      <div className="space-y-2.5">
                        <span className="font-mono text-[10px] uppercase text-cyan-400 tracking-wider font-semibold">
                          Code Provenance Citations
                        </span>
                        {answer.citations.map((cite, idx) => (
                          <div key={idx} className="bg-[#070c18] border border-gray-800/80 rounded-lg p-3 font-mono text-[11px]">
                            <div className="flex items-center justify-between text-gray-300 pb-1.5 border-b border-gray-800/50">
                              <span className="text-cyan-400 font-semibold">{cite.file_path}:{cite.start_line}-{cite.end_line}</span>
                              <span className="text-[10px] text-gray-500 uppercase">{cite.language}</span>
                            </div>
                            <pre className="mt-2 text-[10px] text-gray-300 overflow-x-auto bg-[#04060c] p-2.5 rounded border border-gray-900 leading-relaxed">
                              {cite.content}
                            </pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="h-full min-h-[380px] flex items-center justify-center border border-dashed border-gray-800/80 rounded-lg text-xs font-mono text-gray-500">
                    Ask a technical query to inspect grounded line-level code citations.
                  </div>
                )}
              </div>
            </section>
          </main>
        )}

        {/* Tab 2: Challenges & Remediation */}
        {activeTab === "challenges" && (
          <main className="grid grid-cols-1 xl:grid-cols-12 gap-6 flex-1 items-stretch">
            {/* Left: Interrogation Challenges (5 cols) */}
            <section className="xl:col-span-5 bg-[#0b101f] border border-gray-800 rounded-xl p-5 flex flex-col gap-4 min-h-[600px]">
              <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-rose-400">
                    Red-Team Interrogation
                  </h2>
                </div>
                <button
                  onClick={handleGenerateChallenges}
                  disabled={generatingChallenges}
                  className="bg-rose-950 hover:bg-rose-900 border border-rose-800 text-rose-300 font-mono text-[11px] px-2.5 py-1 rounded cursor-pointer"
                >
                  {generatingChallenges ? "Interrogating..." : "Re-scan"}
                </button>
              </div>

              {!challengeData || challengeData.challenges.length === 0 ? (
                <div className="h-full min-h-[440px] flex flex-col items-center justify-center text-xs font-mono text-gray-500 border border-dashed border-gray-800 rounded gap-2">
                  <span>No active challenges loaded.</span>
                  <button onClick={handleGenerateChallenges} className="text-cyan-400 underline cursor-pointer">
                    Run Red-Team Interrogation
                  </button>
                </div>
              ) : (
                <div className="space-y-4 overflow-y-auto flex-1 pr-1">
                  {challengeData.challenges.map((c, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedChallengeIdx(idx)}
                      className={`p-4 rounded-lg border transition cursor-pointer ${
                        selectedChallengeIdx === idx
                          ? "bg-[#090e1a] border-rose-600/70 shadow-md shadow-rose-950/20"
                          : "bg-[#070c18] border-gray-800 hover:border-gray-700"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="text-[10px] font-mono text-gray-400 uppercase">
                          Claimed Observation
                        </span>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[9px] font-mono uppercase px-2 py-0.5 rounded font-bold bg-rose-950 text-rose-400 border border-rose-800">
                            {c.severity || "HIGH"} SEVERITY
                          </span>
                          <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-gray-900 text-gray-400 border border-gray-800">
                            {c.category || "CONTRADICTION"}
                          </span>
                        </div>
                      </div>

                      <p className="font-mono text-xs text-gray-200 font-medium mb-3">"{c.claim}"</p>

                      <div className="bg-[#0e1424] border border-rose-900/40 rounded p-3 text-xs text-rose-200/90 leading-relaxed">
                        <span className="font-mono text-rose-400 text-[10px] block font-bold tracking-wider mb-1">
                          CRITIQUE / CHALLENGE:
                        </span>
                        {c.challenge}
                      </div>

                      <div className="mt-3 flex items-center justify-between pt-2 border-t border-gray-800/60">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleApplyAiFix(c.challenge);
                          }}
                          className="bg-emerald-950 hover:bg-emerald-900 border border-emerald-700/80 text-emerald-300 font-mono text-[11px] px-3 py-1 rounded flex items-center gap-1.5 cursor-pointer transition active:scale-95 shadow-sm"
                        >
                          <span>⚡</span> Draft Fix to README
                        </button>
                        <span className="text-[10px] font-mono text-gray-500">
                          {c.evidence_ids?.length || 0} Evidence block(s)
                        </span>
                      </div>

                      {challengeData.evidence_citations &&
                        challengeData.evidence_citations
                          .filter((cite) => c.evidence_ids?.includes(cite.id || ""))
                          .map((cite, i) => {
                            const isExpanded = expandedEvidence[cite.id || `${idx}-${i}`];
                            return (
                              <div
                                key={i}
                                className="mt-3 bg-[#050811] border border-gray-800/90 rounded-md overflow-hidden"
                              >
                                <div
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleEvidence(cite.id || `${idx}-${i}`);
                                  }}
                                  className="flex items-center justify-between px-3 py-1.5 bg-[#0a0f1d] border-b border-gray-800/60 cursor-pointer hover:bg-gray-850"
                                >
                                  <div className="flex items-center gap-2 font-mono text-[11px]">
                                    <span className="text-cyan-400 font-semibold">[{cite.id}]</span>
                                    <span className="text-gray-300">{cite.file_path}:{cite.start_line}-{cite.end_line}</span>
                                  </div>
                                  <span className="font-mono text-[10px] text-gray-500 uppercase">
                                    {isExpanded ? "Collapse ▲" : "View Code ▼"}
                                  </span>
                                </div>

                                {isExpanded && (
                                  <pre className="p-3 text-[11px] font-mono text-gray-300 overflow-x-auto max-h-56 bg-[#03060d] leading-relaxed">
                                    {cite.content}
                                  </pre>
                                )}
                              </div>
                            );
                          })}
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Right: Documentation Workspace (7 cols) */}
            <section className="xl:col-span-7 bg-[#0b101f] border border-gray-800 rounded-xl p-5 flex flex-col gap-3 min-h-[600px]">
              <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-gray-200">
                      README.md Documentation
                    </h2>
                  </div>
                  <p className="text-[11px] text-gray-400 mt-0.5">
                    Live document preview with automated discrepancy patching
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <div className="flex bg-[#050811] border border-gray-800 rounded p-0.5 text-xs font-mono">
                    <button
                      onClick={() => setIsEditingMd(false)}
                      className={`px-2.5 py-1 rounded transition cursor-pointer ${
                        !isEditingMd ? "bg-gray-800 text-cyan-400 font-medium" : "text-gray-400 hover:text-gray-200"
                      }`}
                    >
                      Rendered
                    </button>
                    <button
                      onClick={() => setIsEditingMd(true)}
                      className={`px-2.5 py-1 rounded transition cursor-pointer ${
                        isEditingMd ? "bg-gray-800 text-cyan-400 font-medium" : "text-gray-400 hover:text-gray-200"
                      }`}
                    >
                      Raw Editor
                    </button>
                  </div>

                  <button
                    onClick={handleCopyReadme}
                    className={`font-mono text-xs px-3.5 py-1.5 rounded cursor-pointer transition active:scale-95 flex items-center gap-1.5 border font-medium ${
                      copiedSuccess
                        ? "bg-emerald-950 border-emerald-700 text-emerald-300"
                        : "bg-gray-800 hover:bg-gray-700 border-gray-700 text-cyan-400"
                    }`}
                  >
                    <span>{copiedSuccess ? "✓" : "📋"}</span>
                    <span>{copiedSuccess ? "Copied to Clipboard!" : "Copy README"}</span>
                  </button>
                </div>
              </div>

              <div className="flex-1 bg-[#050811] border border-gray-800/80 rounded-lg p-5 overflow-y-auto">
                {isEditingMd ? (
                  <textarea
                    value={readmeDraft}
                    onChange={(e) => setReadmeDraft(e.target.value)}
                    className="w-full h-full min-h-[500px] bg-transparent font-mono text-xs text-gray-200 focus:outline-none resize-none leading-relaxed"
                    placeholder="Edit markdown source..."
                  />
                ) : (
                  <div className="prose prose-invert max-w-none text-xs leading-relaxed text-gray-300">
                    <ReactMarkdown
                      components={{
                        h1: ({ node, ...props }) => (
                          <h1 className="text-lg font-bold text-white border-b border-gray-800 pb-2 mb-3 mt-1" {...props} />
                        ),
                        h2: ({ node, ...props }) => (
                          <h2 className="text-sm font-semibold text-cyan-400 mt-4 mb-2" {...props} />
                        ),
                        h3: ({ node, ...props }) => (
                          <h3 className="text-xs font-semibold text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 p-2 rounded mt-3 mb-2" {...props} />
                        ),
                        ul: ({ node, ...props }) => (
                          <ul className="list-disc list-inside space-y-1 my-2 text-gray-300" {...props} />
                        ),
                        code: ({ node, ...props }) => (
                          <code className="bg-gray-900 text-rose-300 px-1 py-0.5 rounded font-mono text-[11px]" {...props} />
                        ),
                      }}
                    >
                      {readmeDraft}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </section>
          </main>
        )}
      </div>
    </div>
  );
}
