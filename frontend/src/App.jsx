import React, { useState } from "react";
import axios from "axios";
import {
  Upload,
  Activity,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Database,
  ChevronRight,
  Loader2,
  Brain,
  Cpu,
  Layers,
  Network,
  Code,
  Binary,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API_URL = "http://localhost:8000";

const FloatingBackground = () => {
  const icons = [Brain, Cpu, Database, Layers, Network, Code, Binary, Activity];
  const items = Array.from({ length: 30 });

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {items.map((_, i) => {
        const Icon = icons[i % icons.length];
        const size = Math.random() * 30 + 15;
        const duration = Math.random() * 20 + 20;
        const delay = Math.random() * 15;
        const leftPos = Math.random() * 100;

        return (
          <motion.div
            key={i}
            initial={{ y: -100, rotate: 0, opacity: 0 }}
            animate={{
              y: "110vh",
              rotate: 360,
              opacity: [0, 0.4, 0.4, 0],
            }}
            transition={{
              duration: duration,
              repeat: Infinity,
              delay: delay,
              ease: "linear",
            }}
            style={{
              position: "absolute",
              left: `${leftPos}%`,
            }}
          >
            <Icon
              size={size}
              strokeWidth={1}
              className="text-primary/60 drop-shadow-[0_0_8px_rgba(139,92,246,0.3)]"
            />
          </motion.div>
        );
      })}
    </div>
  );
};

function App() {
  const [milestone, setMilestone] = useState("1");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }

    setLoading(true);
    setResults(null);
    setError(null);

    const formData = new FormData();
    formData.append("milestone", milestone);
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResults(response.data.results);
    } catch (err) {
      setError(
        err.response?.data?.detail || "An error occurred during prediction.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-background text-foreground font-sans selection:bg-primary/30 overflow-x-hidden">
      <FloatingBackground />
      {/* Decorative background elements */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]" />
      </div>

      <div className="relative z-10">
        <main className="max-w-5xl mx-auto px-6 py-12">
          <header className="mb-12 text-center">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary text-primary text-xs font-medium mb-4"
            >
              <Activity className="w-3 h-3" />
              <span>AI Model Evaluation Suite</span>
            </motion.div>
            <motion.h1
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-5xl font-bold tracking-tight mb-4"
            >
              Predictive <span className="gradient-text">Analytics</span>{" "}
              Dashboard
            </motion.h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Analyze insights from the{" "}
              <span className="text-foreground font-semibold">
                Stack Overflow Developer Survey
              </span>
              .
            </p>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Controls Panel */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="lg:col-span-4 space-y-6"
            >
              <div className="glass p-6 rounded-2xl border border-border">
                <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                  <Database className="w-5 h-5 text-primary" />
                  Configuration
                </h2>

                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground mb-3 block">
                      Select Milestone
                    </label>
                    <div className="grid grid-cols-2 gap-2 p-1 bg-secondary rounded-lg">
                      <button
                        onClick={() => setMilestone("1")}
                        className={`py-2 px-4 rounded-md text-sm transition-all ${
                          milestone === "1"
                            ? "bg-primary text-primary-foreground shadow-lg"
                            : "hover:bg-background/50"
                        }`}
                      >
                        Regression (M1)
                      </button>
                      <button
                        onClick={() => setMilestone("2")}
                        className={`py-2 px-4 rounded-md text-sm transition-all ${
                          milestone === "2"
                            ? "bg-primary text-primary-foreground shadow-lg"
                            : "hover:bg-background/50"
                        }`}
                      >
                        Classification (M2)
                      </button>
                    </div>
                  </div>

                  <form onSubmit={handleSubmit} className="space-y-4 pt-4">
                    <div>
                      <label className="text-sm font-medium text-muted-foreground mb-3 block">
                        Test Data (CSV)
                      </label>
                      <div className="relative">
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handleFileChange}
                          className="hidden"
                          id="csv-upload"
                        />
                        <label
                          htmlFor="csv-upload"
                          className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
                            file
                              ? "border-primary/50 bg-primary/5"
                              : "border-border hover:border-primary/30"
                          }`}
                        >
                          {file ? (
                            <div className="flex flex-center flex-col items-center gap-2">
                              <CheckCircle className="w-8 h-8 text-primary" />
                              <span className="text-sm font-medium text-center px-4 truncate max-w-[200px]">
                                {file.name}
                              </span>
                            </div>
                          ) : (
                            <div className="flex flex-col items-center gap-2">
                              <Upload className="w-8 h-8 text-muted-foreground" />
                              <span className="text-sm text-muted-foreground">
                                Browse CSV
                              </span>
                            </div>
                          )}
                        </label>
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={loading || !file}
                      className="w-full py-3 bg-primary text-primary-foreground rounded-xl font-semibold flex items-center justify-center gap-2 transition-transform active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-110"
                    >
                      {loading ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <>
                          Run Analysis
                          <ChevronRight className="w-4 h-4" />
                        </>
                      )}
                    </button>
                  </form>

                  {error && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="flex items-center gap-2 p-3 bg-destructive/10 text-destructive text-xs rounded-lg border border-destructive/20"
                    >
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>{error}</span>
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>

            {/* Results Panel */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="lg:col-span-8"
            >
              <div className="glass p-6 rounded-2xl border border-border min-h-[400px]">
                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-xl font-semibold flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-primary" />
                    Model Performance
                  </h2>
                  {results && (
                    <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded">
                      {results.length} Models Tested
                    </span>
                  )}
                </div>

                {!results && !loading && (
                  <div className="h-full flex flex-col items-center justify-center text-muted-foreground py-20">
                    <Activity className="w-12 h-12 mb-4 opacity-20" />
                    <p>Upload data to see evaluation results</p>
                  </div>
                )}

                {loading && (
                  <div className="h-full flex flex-col items-center justify-center py-20">
                    <div className="w-16 h-16 border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-4" />
                    <p className="text-muted-foreground animate-pulse">
                      Processing pipeline...
                    </p>
                  </div>
                )}

                <AnimatePresence>
                  {results && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="space-y-6"
                    >

                      <div className="overflow-hidden rounded-xl border border-border bg-background/50">
                        <table className="w-full text-left text-sm">
                          <thead className="bg-secondary/50 border-b border-border">
                            <tr>
                              <th className="px-4 py-3 font-medium">Model</th>
                              {milestone === "1" ? (
                                <>
                                  <th className="px-4 py-3 font-medium">R²</th>
                                  <th className="px-4 py-3 font-medium">MSE</th>
                                  <th className="px-4 py-3 font-medium">
                                    RMSE
                                  </th>
                                </>
                              ) : (
                                <th className="px-4 py-3 font-medium">
                                  Accuracy
                                </th>
                              )}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {results.map((res, i) => (
                              <motion.tr
                                key={res.Model}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: i * 0.05 }}
                                className="hover:bg-white/[0.02] transition-colors"
                              >
                                <td className="px-4 py-3 font-medium">
                                  {res.Model}
                                </td>
                                {milestone === "1" ? (
                                  <>
                                    <td className="px-4 py-3 text-primary">
                                      {res.R2?.toFixed(4)}
                                    </td>
                                    <td className="px-4 py-3 text-muted-foreground">
                                      {res.MSE?.toFixed(2)}
                                    </td>
                                    <td className="px-4 py-3 text-muted-foreground">
                                      {res.RMSE?.toFixed(2)}
                                    </td>
                                  </>
                                ) : (
                                  <td className="px-4 py-3 text-emerald-400 font-semibold">
                                    {res.Accuracy?.toFixed(4)}
                                  </td>
                                )}
                              </motion.tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
