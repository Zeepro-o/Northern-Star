import axios from "axios";

export const API_BASE = "http://127.0.0.1:8000/api/v1";

export interface Citation {
  id?: string;
  file_path: string;
  start_line: number;
  end_line: number;
  language: string;
  content: string;
}

export interface Claim {
  id: string;
  text: string;
  category?: string;
  source?: string;
  verdict?: "supported" | "refuted" | "unclear";
  verdict_explanation?: string;
  evidence_ids?: string[];
}

export interface AnswerResponse {
  answer: string;
  confidence: string;
  evidence_grounding: string;
  citations: Citation[];
}

export interface DimensionScore {
  name: string;
  score: number;
  weight?: number;
  reasoning?: string;
}

export interface JudgeResult {
  repo_id: string;
  overall_score: number;
  dimensions: DimensionScore[];
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  claim_integrity_summary: Record<string, number>;
  total_claims: number;
  evidence_citations?: Citation[];
}

export interface Challenge {
  id: string;
  claim: string;
  challenge: string;
  category?: string;
  severity?: "high" | "medium" | "low";
  evidence_ids?: string[];
  confidence?: "high" | "medium" | "low";
  explanation?: string;
}

export interface ChallengeResult {
  repo_id: string;
  challenges: Challenge[];
  total_challenges: number;
  high_severity?: number;
  medium_severity?: number;
  low_severity?: number;
  evidence_citations?: Citation[];
}

export const api = {
  async ingestRepo(url: string) {
    const res = await axios.post(`${API_BASE}/repos`, { url });
    return res.data;
  },

  async indexRepo(owner: string, repo: string) {
    const res = await axios.post(`${API_BASE}/repos/${owner}/${repo}/index`);
    return res.data;
  },

  async getClaims(owner: string, repo: string): Promise<Claim[]> {
    const res = await axios.get(`${API_BASE}/repos/${owner}/${repo}/claims`);
    return Array.isArray(res.data) ? res.data : (res.data.claims || []);
  },

  async verifyClaim(owner: string, repo: string, claim_text: string): Promise<Claim> {
    const res = await axios.post(`${API_BASE}/repos/${owner}/${repo}/verify`, {
      claim: claim_text,
      text: claim_text,
    });
    return res.data;
  },

  async askQuestion(owner: string, repo: string, question: string): Promise<AnswerResponse> {
    const res = await axios.post(`${API_BASE}/repos/${owner}/${repo}/ask`, {
      question,
    });
    return res.data;
  },

  async judgeRepo(owner: string, repo: string): Promise<JudgeResult> {
    const res = await axios.get(`${API_BASE}/repos/${owner}/${repo}/judge`);
    return res.data;
  },

  async getChallenges(owner: string, repo: string): Promise<ChallengeResult> {
    const res = await axios.get(`${API_BASE}/repos/${owner}/${repo}/challenges`);
    return res.data;
  }
};
