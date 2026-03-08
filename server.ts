import express from "express";
import { createServer as createViteServer } from "vite";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawn, type ChildProcessWithoutNullStreams } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;
  const trainingProcesses = new Map<string, ChildProcessWithoutNullStreams>();

  app.use(express.json({ limit: "20mb" }));

  // Ensure logs directory exists
  const logsDir = path.join(__dirname, "logs");
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir);
  }
  const runsDir = path.join(__dirname, "backend", "runs");
  if (!fs.existsSync(runsDir)) {
    fs.mkdirSync(runsDir, { recursive: true });
  }

  const readJson = (targetPath: string) => {
    if (!fs.existsSync(targetPath)) return null;
    return JSON.parse(fs.readFileSync(targetPath, "utf8"));
  };

  const buildRunPaths = (runId: string) => {
    const runDir = path.join(runsDir, runId);
    return {
      runDir,
      datasetPath: path.join(runDir, "dataset.json"),
      statusPath: path.join(runDir, "status.json"),
      metricsPath: path.join(runDir, "metrics.json"),
      lossCurvePath: path.join(runDir, "loss_curve.json"),
      predictionsPath: path.join(runDir, "predictions.json"),
      checkpointPath: path.join(runDir, "model.pt"),
      stderrPath: path.join(runDir, "stderr.log"),
      stdoutPath: path.join(runDir, "stdout.log"),
    };
  };

  // API Routes
  app.post("/api/logs/training", (req, res) => {
    const { timestamp, params, metrics, status } = req.body;
    const logEntry = {
      timestamp: timestamp || new Date().toISOString(),
      type: "TRAINING",
      params,
      metrics,
      status
    };
    
    fs.appendFileSync(
      path.join(logsDir, "training.log"),
      JSON.stringify(logEntry) + "\n"
    );
    
    res.json({ success: true });
  });

  app.post("/api/logs/evaluation", (req, res) => {
    const { timestamp, metrics, stability, params } = req.body;
    const logEntry = {
      timestamp: timestamp || new Date().toISOString(),
      type: "EVALUATION",
      metrics,
      stability,
      params
    };
    
    fs.appendFileSync(
      path.join(logsDir, "evaluation.log"),
      JSON.stringify(logEntry) + "\n"
    );
    
    res.json({ success: true });
  });

  app.get("/api/logs", (req, res) => {
    try {
      const trainingLogs = fs.existsSync(path.join(logsDir, "training.log")) 
        ? fs.readFileSync(path.join(logsDir, "training.log"), "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l))
        : [];
      const evaluationLogs = fs.existsSync(path.join(logsDir, "evaluation.log"))
        ? fs.readFileSync(path.join(logsDir, "evaluation.log"), "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l))
        : [];
        
      res.json({ training: trainingLogs, evaluation: evaluationLogs });
    } catch (error) {
      res.status(500).json({ error: "Failed to read logs" });
    }
  });

  app.get("/api/datasets/reversal-only", (_req, res) => {
    try {
      const datasetPath = path.join(__dirname, "backend", "runs", "reversal_real_csv_dataset.json");
      if (!fs.existsSync(datasetPath)) {
        res.status(404).json({ error: "Reversal-only dataset not found." });
        return;
      }
      res.json({
        name: "reversal_only",
        source: "csv",
        profile: "reversal_only",
        dataset: JSON.parse(fs.readFileSync(datasetPath, "utf8")),
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to load reversal-only dataset." });
    }
  });

  app.post("/api/lnn/train", (req, res) => {
    try {
      const { dataset, params, systemParams } = req.body;
      if (!Array.isArray(dataset) || dataset.length < 8) {
        res.status(400).json({ error: "Dataset is missing or too short." });
        return;
      }

      const runId = `run_${Date.now()}`;
      const { runDir, datasetPath, statusPath, stderrPath, stdoutPath } = buildRunPaths(runId);
      fs.mkdirSync(runDir, { recursive: true });
      fs.writeFileSync(datasetPath, JSON.stringify(dataset, null, 2));
      fs.writeFileSync(
        statusPath,
        JSON.stringify(
          {
            state: "queued",
            progress: 0,
            epoch: 0,
            totalEpochs: params?.epochs ?? 100,
            message: "Queued for training",
            lossCurve: [],
          },
          null,
          2
        )
      );

      const pythonArgs = [
        "run",
        "-n",
        "nerd_newton",
        "python",
        path.join(__dirname, "backend", "train_lnn.py"),
        "--dataset",
        datasetPath,
        "--run-dir",
        runDir,
        "--hidden-units",
        String(params?.hiddenUnits ?? 64),
        "--learning-rate",
        String(params?.learningRate ?? 0.001),
        "--window-size",
        String(params?.windowSize ?? 50),
        "--epochs",
        String(params?.epochs ?? 100),
        "--batch-size",
        String(params?.batchSize ?? 1024),
        "--pos-loss-weight",
        String(params?.posLossWeight ?? 1.0),
        "--vel-loss-weight",
        String(params?.velLossWeight ?? 1.0),
        "--torque-loss-weight",
        String(params?.torqueLossWeight ?? 0.1),
        "--one-step-loss-weight",
        String(params?.oneStepLossWeight ?? 1.0),
        "--rollout-loss-weight",
        String(params?.rolloutLossWeight ?? 0.0),
        "--train-rollout-steps",
        String(params?.trainRolloutSteps ?? 10),
        "--controller-kp",
        String(systemParams?.kp ?? 1.0),
        "--controller-kd",
        String(systemParams?.kd ?? 0.01),
        "--precision",
        String(params?.precision ?? "bf16"),
        "--device",
        "cuda",
      ];
      if (params?.compileModel) {
        pythonArgs.splice(pythonArgs.length - 2, 0, "--compile-model");
      }

      const child = spawn("conda", pythonArgs, {
        cwd: __dirname,
        stdio: "pipe",
      });
      trainingProcesses.set(runId, child);

      child.stdout.on("data", (chunk) => {
        fs.appendFileSync(stdoutPath, chunk.toString());
      });
      child.stderr.on("data", (chunk) => {
        const text = chunk.toString();
        fs.appendFileSync(stderrPath, text);
        console.error(`[${runId}]`, text);
      });

      child.on("close", (code) => {
        trainingProcesses.delete(runId);
        if (code !== 0) {
          fs.writeFileSync(
            statusPath,
            JSON.stringify(
              {
                state: "failed",
                progress: 100,
                epoch: 0,
                totalEpochs: params?.epochs ?? 100,
                message: `Training failed with exit code ${code}`,
                errorLog: fs.existsSync(stderrPath) ? stderrPath : null,
                lossCurve: [],
              },
              null,
              2
            )
          );
        }
      });

      res.json({ success: true, runId });
    } catch (error) {
      res.status(500).json({ error: "Failed to start LNN training." });
    }
  });

  app.get("/api/lnn/train/:runId/status", (req, res) => {
    const { statusPath, lossCurvePath, stderrPath, stdoutPath } = buildRunPaths(req.params.runId);
    const status = readJson(statusPath);
    if (!status) {
      res.status(404).json({ error: "Run not found." });
      return;
    }
    const lossCurve = readJson(lossCurvePath);
    res.json({
      ...status,
      lossCurve: Array.isArray(lossCurve) && lossCurve.length > 0 ? lossCurve : status.lossCurve ?? [],
      stderr: fs.existsSync(stderrPath) ? fs.readFileSync(stderrPath, "utf8") : "",
      stdout: fs.existsSync(stdoutPath) ? fs.readFileSync(stdoutPath, "utf8") : "",
    });
  });

  app.post("/api/lnn/train/:runId/stop", (req, res) => {
    try {
      const { runId } = req.params;
      const { statusPath, stderrPath } = buildRunPaths(runId);
      const child = trainingProcesses.get(runId);
      const existingStatus = readJson(statusPath);

      if (!child && !existingStatus) {
        res.status(404).json({ error: "Run not found." });
        return;
      }

      if (child && !child.killed) {
        try {
          child.kill("SIGTERM");
        } catch (error) {
          console.error(`Failed to stop run ${runId}`, error);
        }
      }
      trainingProcesses.delete(runId);

      fs.writeFileSync(
        statusPath,
        JSON.stringify(
          {
            state: "stopped",
            progress: existingStatus?.progress ?? 0,
            epoch: existingStatus?.epoch ?? 0,
            totalEpochs: existingStatus?.totalEpochs ?? 0,
            message: "Training stopped by user",
            errorLog: fs.existsSync(stderrPath) ? stderrPath : null,
            lossCurve: Array.isArray(existingStatus?.lossCurve) ? existingStatus.lossCurve : [],
          },
          null,
          2
        )
      );

      res.json({ success: true, runId, state: "stopped" });
    } catch (error) {
      res.status(500).json({ error: "Failed to stop LNN training." });
    }
  });

  app.post("/api/lnn/eval/:runId", (req, res) => {
    try {
      const horizonSteps = Number(req.body?.horizonSteps);
      const evalMode = req.body?.mode === "one_step" ? "one_step" : "rollout";
      const { runDir, datasetPath, checkpointPath, stderrPath, stdoutPath } = buildRunPaths(req.params.runId);
      if (!fs.existsSync(checkpointPath) || !fs.existsSync(datasetPath)) {
        res.status(404).json({ error: "Missing model checkpoint or dataset." });
        return;
      }

      const evalArgs = [
        "run",
        "-n",
        "nerd_newton",
        "python",
        path.join(__dirname, "backend", "eval_lnn.py"),
        "--dataset",
        datasetPath,
        "--run-dir",
        runDir,
        "--eval-mode",
        evalMode,
        "--device",
        "cuda",
      ];
      if (Number.isFinite(horizonSteps)) {
        evalArgs.push("--horizon-steps", String(Math.max(0, Math.round(horizonSteps))));
      }

      const child = spawn("conda", evalArgs, { cwd: __dirname, stdio: "pipe" });
      let stderr = "";
      let stdout = "";

      child.stdout.on("data", (chunk) => {
        const text = chunk.toString();
        stdout += text;
      });

      child.stderr.on("data", (chunk) => {
        const text = chunk.toString();
        stderr += text;
        console.error(`[eval:${req.params.runId}]`, text);
      });

      child.on("close", (code) => {
        if (code !== 0) {
          if (stdout) {
            fs.appendFileSync(stdoutPath, `\n[EVAL ${evalMode}]\n${stdout}`);
          }
          if (stderr) {
            fs.appendFileSync(stderrPath, `\n[EVAL ${evalMode}]\n${stderr}`);
          }
          console.error(`Evaluation failed for ${req.params.runId}`);
          if (!res.headersSent) {
            res.status(500).json({ error: "Evaluation failed.", mode: evalMode });
          }
          return;
        }

        if (stdout) {
          fs.appendFileSync(stdoutPath, `\n[EVAL ${evalMode}]\n${stdout}`);
        }
        if (stderr) {
          fs.appendFileSync(stderrPath, `\n[EVAL ${evalMode}]\n${stderr}`);
        }
        if (!res.headersSent) {
          res.json({ success: true, mode: evalMode });
        }
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to run evaluation." });
    }
  });

  app.get("/api/lnn/runs/:runId/results", (req, res) => {
    const { metricsPath, lossCurvePath, predictionsPath, statusPath } = buildRunPaths(req.params.runId);
    const metrics = readJson(metricsPath);
    const lossCurve = readJson(lossCurvePath);
    const predictions = readJson(predictionsPath);
    const status = readJson(statusPath);

    if (!status) {
      res.status(404).json({ error: "Run not found." });
      return;
    }

    res.json({
      runId: req.params.runId,
      status,
      metrics: metrics ?? null,
      lossCurve: Array.isArray(lossCurve) ? lossCurve : [],
      predictions: Array.isArray(predictions) ? predictions : [],
    });
  });

  app.get("/api/lnn/runs/:runId/preview", (req, res) => {
    const { metricsPath, predictionsPath, statusPath } = buildRunPaths(req.params.runId);
    const metrics = readJson(metricsPath);
    const predictions = readJson(predictionsPath);
    const status = readJson(statusPath);

    if (!status) {
      res.status(404).json({ error: "Run not found." });
      return;
    }

    const limit = Math.max(200, Math.min(Number(req.query.limit ?? 3000), 5000));
    const rows = Array.isArray(predictions) ? predictions : [];
    let previewRows = rows;
    if (rows.length > limit) {
      const stride = Math.ceil(rows.length / limit);
      previewRows = rows.filter((_, index) => index % stride === 0);
      if (previewRows[previewRows.length - 1] !== rows[rows.length - 1]) {
        previewRows.push(rows[rows.length - 1]);
      }
    }

    res.json({
      runId: req.params.runId,
      status,
      metrics: metrics ?? null,
      predictions: previewRows,
      fullCount: rows.length,
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, "dist")));
    app.get("*", (req, res) => {
      res.sendFile(path.join(__dirname, "dist", "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
