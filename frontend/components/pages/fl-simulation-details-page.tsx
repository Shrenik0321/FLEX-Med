"use client";

import { ArrowLeft, Users } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
// import { toast } from "sonner"; // Commented out for static data
// import { API_BASE_PATH } from "@/utils"; // Commented out for static data
import {
  FLSimulation,
  parseMetrics,
  FLSimulationMetrics,
} from "@/types/fl-simulation";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import EvalCard from "@/components/ui/eval-card";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  ChartLegend,
);

interface FLSimulationDetailsPageProps {
  simulationId: number;
  simulationName: string;
  onBack: () => void;
}

interface Client {
  id: number;
  client_name: string;
  model_type: string;
  metrics: string;
}

interface ClientMetrics {
  global: {
    pre_fl: any;
    post_fl: any;
    improvement: any;
  };
  rounds: Array<{
    round: number;
    validation: {
      loss: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      evaluated_at: string;
    };
  }>;
}

// ============================================================================
// TEMPORARY: Static data from cnmc_data.json for testing/demo purposes
// To revert to API:
//   1. Remove this TEMP_CNMC_DATA constant
//   2. Uncomment the API code in the useEffect (lines ~677-690)
//   3. Remove the mock data code (lines ~599-673)
// ============================================================================
const TEMP_CNMC_DATA = [
  {
    id: 26,
    created_at: "2026-01-14T16:46:50.703429+00:00",
    client_name: "Asiri-resnet-allidb2",
    model_type: "resnet18",
    dataset_path:
      "/content/drive/MyDrive/College/FLEX-Med/cnmc_datasets/cnmc_client_0",
    status: "Training",
    has_local_data: true,
    model_path: "/content/drive/MyDrive/College/models/Asiri-resnet-allidb2.pt",
    metrics: {
      global: {
        pre_fl: {
          loss: 0.691839,
          accuracy: 0.498403,
          precision: 0.0,
          recall: 0.0,
          f1_score: 0.0,
          specificity: 0.996805,
          roc_auc: 0.41383,
          leukemia_accuracy: 0.0,
          healthy_accuracy: 0.996805,
          class_gap: 0.996805,
          confusion_matrix: {
            TP: 0,
            FP: 2,
            FN: 626,
            TN: 624,
          },
          num_samples: 1252,
          num_leukemia_samples: 626,
          num_healthy_samples: 626,
          dataset: "public_test",
          evaluation_type: "global",
          evaluated_at: "2026-01-24T21:51:34.277660",
        },
        post_fl: {
          loss: 0.590043,
          accuracy: 0.726038,
          precision: 0.67362,
          recall: 0.876997,
          f1_score: 0.761971,
          specificity: 0.57508,
          roc_auc: 0.192099,
          leukemia_accuracy: 0.876997,
          healthy_accuracy: 0.57508,
          class_gap: 0.301917,
          confusion_matrix: {
            TP: 549,
            FP: 266,
            FN: 77,
            TN: 360,
          },
          num_samples: 1252,
          num_leukemia_samples: 626,
          num_healthy_samples: 626,
          dataset: "public_test",
          evaluation_type: "global",
          evaluated_at: "2026-01-25T03:39:13.631859",
        },
        improvement: {
          accuracy: 0.227635,
          loss: -0.101796,
          precision: 0.67362,
          recall: 0.876997,
          f1_score: 0.761971,
          class_gap: -0.694888,
          leukemia_accuracy: 0.876997,
          healthy_accuracy: -0.421725,
        },
      },
      rounds: [
        {
          round: 1,
          validation: {
            loss: 0.48867629276240904,
            accuracy: 0.8270106221547799,
            precision: 0.7064220183486238,
            recall: 0.48427672955974843,
            f1_score: 0.5746268656716418,
            class_gap: 0.4517232704402516,
            leukemia_accuracy: 0.48427672955974843,
            healthy_accuracy: 0.936,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-24T22:17:39.811246",
          },
        },
        {
          round: 2,
          validation: {
            loss: 0.5377411655120676,
            accuracy: 0.8088012139605463,
            precision: 0.610738255033557,
            recall: 0.5723270440251572,
            f1_score: 0.5909090909090909,
            class_gap: 0.3116729559748428,
            leukemia_accuracy: 0.5723270440251572,
            healthy_accuracy: 0.884,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-24T22:52:45.674457",
          },
        },
        {
          round: 3,
          validation: {
            loss: 0.5496897764379591,
            accuracy: 0.7966616084977238,
            precision: 0.5757575757575758,
            recall: 0.5974842767295597,
            f1_score: 0.5864197530864198,
            class_gap: 0.26251572327044026,
            leukemia_accuracy: 0.5974842767295597,
            healthy_accuracy: 0.86,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-24T23:28:12.681500",
          },
        },
        {
          round: 4,
          validation: {
            loss: 0.5449337712368943,
            accuracy: 0.8057663125948407,
            precision: 0.6115107913669064,
            recall: 0.5345911949685535,
            f1_score: 0.5704697986577181,
            class_gap: 0.35740880503144656,
            leukemia_accuracy: 0.5345911949685535,
            healthy_accuracy: 0.892,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T00:03:55.002620",
          },
        },
        {
          round: 5,
          validation: {
            loss: 0.550116348972089,
            accuracy: 0.7921092564491654,
            precision: 0.56875,
            recall: 0.5723270440251572,
            f1_score: 0.5705329153605015,
            class_gap: 0.28967295597484277,
            leukemia_accuracy: 0.5723270440251572,
            healthy_accuracy: 0.862,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T00:39:15.007140",
          },
        },
        {
          round: 6,
          validation: {
            loss: 0.5422606685273745,
            accuracy: 0.8179059180576631,
            precision: 0.6344827586206897,
            recall: 0.5786163522012578,
            f1_score: 0.6052631578947368,
            class_gap: 0.3153836477987422,
            leukemia_accuracy: 0.5786163522012578,
            healthy_accuracy: 0.894,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T01:14:38.147588",
          },
        },
        {
          round: 7,
          validation: {
            loss: 0.5424498699503711,
            accuracy: 0.8209408194233687,
            precision: 0.6496350364963503,
            recall: 0.559748427672956,
            f1_score: 0.6013513513513513,
            class_gap: 0.34425157232704406,
            leukemia_accuracy: 0.559748427672956,
            healthy_accuracy: 0.904,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T01:49:40.495015",
          },
        },
        {
          round: 8,
          validation: {
            loss: 0.5470855595128489,
            accuracy: 0.8133535660091047,
            precision: 0.6232876712328768,
            recall: 0.5723270440251572,
            f1_score: 0.5967213114754099,
            class_gap: 0.3176729559748428,
            leukemia_accuracy: 0.5723270440251572,
            healthy_accuracy: 0.89,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T02:25:35.214967",
          },
        },
        {
          round: 9,
          validation: {
            loss: 0.5572788354478584,
            accuracy: 0.7890743550834598,
            precision: 0.5602409638554217,
            recall: 0.5849056603773585,
            f1_score: 0.5723076923076923,
            class_gap: 0.2690943396226415,
            leukemia_accuracy: 0.5849056603773585,
            healthy_accuracy: 0.854,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T03:01:43.647691",
          },
        },
        {
          round: 10,
          validation: {
            loss: 0.5504483489237715,
            accuracy: 0.7966616084977238,
            precision: 0.5827814569536424,
            recall: 0.5534591194968553,
            f1_score: 0.567741935483871,
            class_gap: 0.32054088050314467,
            leukemia_accuracy: 0.5534591194968553,
            healthy_accuracy: 0.874,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T03:37:25.604710",
          },
        },
      ],
    },
  },
  {
    id: 27,
    created_at: "2026-01-14T16:47:52.358916+00:00",
    client_name: "Durdans-mobilenet-cnmc",
    model_type: "mobilenet_v2",
    dataset_path:
      "/content/drive/MyDrive/College/FLEX-Med/cnmc_datasets/cnmc_client_1",
    status: "Training",
    has_local_data: true,
    model_path:
      "/content/drive/MyDrive/College/models/Durdans-mobilenet-cnmc.pt",
    metrics: {
      global: {
        pre_fl: {
          loss: 0.692989,
          accuracy: 0.5,
          precision: 0.0,
          recall: 0.0,
          f1_score: 0.0,
          specificity: 1.0,
          roc_auc: 0.5,
          leukemia_accuracy: 0.0,
          healthy_accuracy: 1.0,
          class_gap: 1.0,
          confusion_matrix: {
            TP: 0,
            FP: 0,
            FN: 626,
            TN: 626,
          },
          num_samples: 1252,
          num_leukemia_samples: 626,
          num_healthy_samples: 626,
          dataset: "public_test",
          evaluation_type: "global",
          evaluated_at: "2026-01-24T21:52:01.545398",
        },
        post_fl: {
          loss: 0.589058,
          accuracy: 0.726837,
          precision: 0.676179,
          recall: 0.870607,
          f1_score: 0.761173,
          specificity: 0.583067,
          roc_auc: 0.193056,
          leukemia_accuracy: 0.870607,
          healthy_accuracy: 0.583067,
          class_gap: 0.28754,
          confusion_matrix: {
            TP: 545,
            FP: 261,
            FN: 81,
            TN: 365,
          },
          num_samples: 1252,
          num_leukemia_samples: 626,
          num_healthy_samples: 626,
          dataset: "public_test",
          evaluation_type: "global",
          evaluated_at: "2026-01-25T03:40:01.443686",
        },
        improvement: {
          accuracy: 0.226837,
          loss: -0.103931,
          precision: 0.676179,
          recall: 0.870607,
          f1_score: 0.761173,
          class_gap: -0.71246,
          leukemia_accuracy: 0.870607,
          healthy_accuracy: -0.416933,
        },
      },
      rounds: [
        {
          round: 1,
          validation: {
            loss: 0.5977774697478878,
            accuracy: 0.7723823975720789,
            precision: 0.5540540540540541,
            recall: 0.4939759036144578,
            f1_score: 0.5222929936305732,
            class_gap: 0.3721498570346294,
            leukemia_accuracy: 0.4939759036144578,
            healthy_accuracy: 0.8661257606490872,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-24T22:18:25.991933",
          },
        },
        {
          round: 2,
          validation: {
            loss: 0.5551386596762176,
            accuracy: 0.795144157814871,
            precision: 0.6099290780141844,
            recall: 0.5180722891566265,
            f1_score: 0.5602605863192183,
            class_gap: 0.3703658447176128,
            leukemia_accuracy: 0.5180722891566265,
            healthy_accuracy: 0.8884381338742393,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-24T22:53:43.246063",
          },
        },
        {
          round: 3,
          validation: {
            loss: 0.5624740644724127,
            accuracy: 0.787556904400607,
            precision: 0.5928571428571429,
            recall: 0.5,
            f1_score: 0.5424836601307189,
            class_gap: 0.38438133874239355,
            leukemia_accuracy: 0.5,
            healthy_accuracy: 0.8843813387423936,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-24T23:29:09.301009",
          },
        },
        {
          round: 4,
          validation: {
            loss: 0.5576311082145332,
            accuracy: 0.7905918057663126,
            precision: 0.5921052631578947,
            recall: 0.5421686746987951,
            f1_score: 0.5660377358490566,
            class_gap: 0.3320706762139838,
            leukemia_accuracy: 0.5421686746987951,
            healthy_accuracy: 0.8742393509127789,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T00:04:55.787536",
          },
        },
        {
          round: 5,
          validation: {
            loss: 0.5562280803602274,
            accuracy: 0.8012139605462822,
            precision: 0.6223776223776224,
            recall: 0.536144578313253,
            f1_score: 0.5760517799352751,
            class_gap: 0.3543219531269093,
            leukemia_accuracy: 0.536144578313253,
            healthy_accuracy: 0.8904665314401623,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T00:40:15.748908",
          },
        },
        {
          round: 6,
          validation: {
            loss: 0.5590003743338114,
            accuracy: 0.7905918057663126,
            precision: 0.5886075949367089,
            recall: 0.5602409638554217,
            f1_score: 0.5740740740740741,
            class_gap: 0.3079131943595885,
            leukemia_accuracy: 0.5602409638554217,
            healthy_accuracy: 0.8681541582150102,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T01:15:28.132013",
          },
        },
        {
          round: 7,
          validation: {
            loss: 0.5581234937734416,
            accuracy: 0.7996965098634294,
            precision: 0.6118421052631579,
            recall: 0.5602409638554217,
            f1_score: 0.5849056603773585,
            class_gap: 0.320083579755126,
            leukemia_accuracy: 0.5602409638554217,
            healthy_accuracy: 0.8803245436105477,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T01:50:29.208780",
          },
        },
        {
          round: 8,
          validation: {
            loss: 0.5590739507957366,
            accuracy: 0.8042488619119879,
            precision: 0.6225165562913907,
            recall: 0.5662650602409639,
            f1_score: 0.5930599369085173,
            class_gap: 0.31811627850142965,
            leukemia_accuracy: 0.5662650602409639,
            healthy_accuracy: 0.8843813387423936,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T02:26:28.760697",
          },
        },
        {
          round: 9,
          validation: {
            loss: 0.5585906929781658,
            accuracy: 0.7890743550834598,
            precision: 0.5894039735099338,
            recall: 0.536144578313253,
            f1_score: 0.5615141955835962,
            class_gap: 0.3380947725995259,
            leukemia_accuracy: 0.536144578313253,
            healthy_accuracy: 0.8742393509127789,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T03:02:35.174887",
          },
        },
        {
          round: 10,
          validation: {
            loss: 0.5597584334418336,
            accuracy: 0.7936267071320182,
            precision: 0.6041666666666666,
            recall: 0.5240963855421686,
            f1_score: 0.5612903225806452,
            class_gap: 0.3602849532002249,
            leukemia_accuracy: 0.5240963855421686,
            healthy_accuracy: 0.8843813387423936,
            num_samples: 659,
            dataset: "validation",
            evaluation_type: "per_round",
            evaluated_at: "2026-01-25T03:38:17.458896",
          },
        },
      ],
    },
  },
];

// Color palette for different clients
const CLIENT_COLORS = [
  "#B80028", // Primary red
  "#3b82f6", // Blue
  "#10b981", // Green
  "#f59e0b", // Orange
  "#8b5cf6", // Purple
  "#ec4899", // Pink
  "#14b8a6", // Teal
  "#f97316", // Orange-red
];

/**
 * SMART Y-AXIS SCALING IMPLEMENTATION
 * ====================================
 *
 * This implementation mirrors the Python fl_evaluation.py smart scaling logic:
 *
 * Key Features:
 * 1. Dynamic range adjustment based on actual data values
 * 2. 15% padding for visual clarity (configurable)
 * 3. Metric-specific optimization:
 *    - Bounded metrics (accuracy, precision, recall, f1): 0-1 range with intelligent scaling
 *    - Unbounded metrics (loss): Open range with padding
 * 4. Minimum range enforcement to prevent flat graphs
 * 5. Pre-FL baseline included as Round 0 (matches Python's extract_validation_series)
 *
 * Changes from original implementation:
 * - Fixed round labeling: Now shows "Round 0, Round 1, ..." instead of "Round -1, Round 0, ..."
 * - Added Pre-FL baseline as Round 0 for all metrics
 * - Replaced fixed y-axis (0-1) with smart dynamic scaling
 * - Per-metric chart options instead of one-size-fits-all
 *
 * Similar to Weights & Biases auto-scaling behavior.
 */

// Smart y-axis scaling function (based on fl_evaluation.py)
function smartYLimit(
  values: number[],
  metricType: "bounded" | "unbounded",
  padding: number = 0.15,
): { min: number; max: number } {
  if (!values || values.length === 0) {
    return { min: 0, max: 1 };
  }

  // Filter out null/undefined/NaN values
  const validValues = values.filter((v) => v != null && !isNaN(v));
  if (validValues.length === 0) {
    return { min: 0, max: 1 };
  }

  const minVal = Math.min(...validValues);
  const maxVal = Math.max(...validValues);
  const valueRange = maxVal - minVal;

  if (metricType === "bounded") {
    // For metrics bounded between 0 and 1 (accuracy, precision, recall, f1)

    // If range is very small, ensure minimum visibility
    if (valueRange < 0.05) {
      const center = (maxVal + minVal) / 2;
      const yMin = Math.max(0, center - 0.05);
      const yMax = Math.min(1.0, center + 0.05);
      return { min: yMin, max: yMax };
    }

    // Add padding to the range
    const padAmount = valueRange * padding;

    // For high values (>0.6), we can start higher than 0
    let yMin: number;
    if (minVal > 0.6) {
      yMin = Math.max(0, minVal - padAmount);
    } else if (minVal > 0.3) {
      yMin = Math.max(0, minVal - padAmount * 1.5);
    } else {
      yMin = 0;
    }

    // Cap at 1.0 but add padding
    const yMax = Math.min(1.0, maxVal + padAmount);

    // Ensure we have at least 10% range for clarity
    if (yMax - yMin < 0.1) {
      const center = (yMax + yMin) / 2;
      return {
        min: Math.max(0, center - 0.05),
        max: Math.min(1.0, center + 0.05),
      };
    }

    return { min: yMin, max: yMax };
  } else {
    // For unbounded metrics (loss)
    if (valueRange < 0.01) {
      const center = (maxVal + minVal) / 2;
      return {
        min: Math.max(0, center - 0.01),
        max: center + 0.01,
      };
    }

    const padAmount = valueRange * padding;
    return {
      min: Math.max(0, minVal - padAmount),
      max: maxVal + padAmount,
    };
  }
}

export default function FLSimulationDetailsPage({
  simulationId,
  simulationName,
  onBack,
}: FLSimulationDetailsPageProps) {
  const [simulation, setSimulation] = useState<FLSimulation | null>(null);
  const [clients, setClients] = useState<Client[]>([]);
  const [selectedClient, setSelectedClient] = useState<string>("all");
  // const [isLoading, setIsLoading] = useState(true); // Commented out for static data

  // Fetch simulation and client data
  useEffect(() => {
    const fetchData = () => {
      try {
        // setIsLoading(true); // Commented out for static data

        // TEMPORARY: Using static data from cnmc_data.json instead of API
        // Convert the static data to match the expected format
        const mockSimulation: FLSimulation = {
          id: simulationId,
          client_ids: TEMP_CNMC_DATA.map((c) => c.id),
          status: "completed",
          configs: {
            num_server_rounds: 10,
            fraction_train: 1.0,
            fraction_evaluate: 1.0,
            local_epochs: 5,
            lr: 0.001,
            lr_decay: 0.1,
            distill_lr: 0.0005,
            distill_epochs: 3,
            temperature: 3.0,
            batch_size: 32,
          },
          metrics: JSON.stringify({
            rounds: TEMP_CNMC_DATA[0].metrics.rounds.map((round, idx) => ({
              round: round.round,
              avg_accuracy:
                TEMP_CNMC_DATA.reduce(
                  (sum, client) =>
                    sum +
                    (client.metrics.rounds[idx]?.validation.accuracy || 0),
                  0,
                ) / TEMP_CNMC_DATA.length,
              avg_precision:
                TEMP_CNMC_DATA.reduce(
                  (sum, client) =>
                    sum +
                    (client.metrics.rounds[idx]?.validation.precision || 0),
                  0,
                ) / TEMP_CNMC_DATA.length,
              avg_recall:
                TEMP_CNMC_DATA.reduce(
                  (sum, client) =>
                    sum + (client.metrics.rounds[idx]?.validation.recall || 0),
                  0,
                ) / TEMP_CNMC_DATA.length,
              avg_f1:
                TEMP_CNMC_DATA.reduce(
                  (sum, client) =>
                    sum +
                    (client.metrics.rounds[idx]?.validation.f1_score || 0),
                  0,
                ) / TEMP_CNMC_DATA.length,
              avg_loss:
                TEMP_CNMC_DATA.reduce(
                  (sum, client) =>
                    sum + (client.metrics.rounds[idx]?.validation.loss || 0),
                  0,
                ) / TEMP_CNMC_DATA.length,
              num_clients_trained: TEMP_CNMC_DATA.length,
              timestamp:
                TEMP_CNMC_DATA[0].metrics.rounds[idx]?.validation
                  .evaluated_at || "",
            })),
            aggregate: {
              pre_fl: {
                avg_accuracy:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.pre_fl.accuracy,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_loss:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.pre_fl.loss,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_precision:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.pre_fl.precision,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_recall:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.pre_fl.recall,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_f1:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.pre_fl.f1_score,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                std_accuracy: 0,
                num_clients: TEMP_CNMC_DATA.length,
              },
              post_fl: {
                avg_accuracy:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.post_fl.accuracy,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_loss:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.post_fl.loss,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_precision:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.post_fl.precision,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_recall:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.post_fl.recall,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_f1:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.post_fl.f1_score,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                std_accuracy: 0,
                num_clients: TEMP_CNMC_DATA.length,
              },
              improvement: {
                avg_accuracy:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.improvement.accuracy,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_loss:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.improvement.loss,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_precision:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.improvement.precision,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_recall:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.improvement.recall,
                    0,
                  ) / TEMP_CNMC_DATA.length,
                avg_f1:
                  TEMP_CNMC_DATA.reduce(
                    (sum, c) => sum + c.metrics.global.improvement.f1_score,
                    0,
                  ) / TEMP_CNMC_DATA.length,
              },
            },
            best_round: {
              round: 1,
              avg_accuracy: Math.max(
                ...TEMP_CNMC_DATA[0].metrics.rounds.map(
                  (r) => r.validation.accuracy,
                ),
              ),
            },
            total_rounds_completed: 10,
            total_clients: TEMP_CNMC_DATA.length,
          }),
          created_at: TEMP_CNMC_DATA[0].created_at,
        };
        setSimulation(mockSimulation);

        // Convert static clients to match the expected Client interface
        const clientsData = TEMP_CNMC_DATA.map((client) => ({
          id: client.id,
          client_name: client.client_name,
          model_type: client.model_type,
          metrics: JSON.stringify(client.metrics), // Convert object to JSON string
        }));
        setClients(clientsData);

        // Original API-based implementation (commented out for temporary use)
        // const simResponse = await fetch(
        //   `${API_BASE_PATH}/fl_simulations/${simulationId}`,
        // );
        // if (!simResponse.ok) throw new Error("Failed to fetch simulation");
        // const simData: FLSimulation = await simResponse.json();
        // setSimulation(simData);

        // const clientIds = simData.client_ids;
        // const clientPromises = clientIds.map((id) =>
        //   fetch(`${API_BASE_PATH}/clients/${id}`).then((res) => res.json()),
        // );
        // const clientsData = await Promise.all(clientPromises);
        // setClients(clientsData);
      } catch (error) {
        console.error("Error loading static data:", error);
        // toast.error("Failed to load simulation data"); // Commented out for static data
      }
      // finally {
      //   setIsLoading(false); // Commented out for static data
      // }
    };

    fetchData();
  }, [simulationId, simulationName]);

  // Parse simulation metrics
  const simulationMetrics = simulation
    ? parseMetrics(simulation.metrics)
    : null;

  // Debug logging
  useEffect(() => {
    if (simulation) {
      console.log("Raw metrics string:", simulation.metrics);
      console.log("Parsed simulationMetrics:", simulationMetrics);
      console.log("Has rounds?", simulationMetrics?.rounds?.length);
    }
  }, [simulation, simulationMetrics]);

  // Parse client metrics
  const clientMetricsMap = new Map<number, ClientMetrics>();
  clients.forEach((client) => {
    try {
      const metrics = JSON.parse(client.metrics) as ClientMetrics;
      clientMetricsMap.set(client.id, metrics);
    } catch (e) {
      console.error(`Failed to parse metrics for client ${client.id}`);
    }
  });

  // Helper function to collect all values for a metric (for smart scaling)
  const collectAllValues = (
    metricKey: keyof ClientMetrics["rounds"][0]["validation"],
  ): number[] => {
    const allValues: number[] = [];

    if (selectedClient === "all") {
      // Collect from all clients
      clients.forEach((client) => {
        const metrics = clientMetricsMap.get(client.id);
        if (metrics) {
          // Include pre-FL baseline
          const preFl = metrics.global?.pre_fl?.[metricKey];
          if (preFl != null) {
            allValues.push(preFl as number);
          }

          // Include validation rounds
          metrics.rounds?.forEach((round) => {
            const value = round.validation[metricKey];
            if (value != null) {
              allValues.push(value as number);
            }
          });

          // Include post-FL
          const postFl = metrics.global?.post_fl?.[metricKey];
          if (postFl != null) {
            allValues.push(postFl as number);
          }
        }
      });
    } else {
      // Collect from selected client
      const clientId = parseInt(selectedClient);
      const metrics = clientMetricsMap.get(clientId);
      if (metrics) {
        const preFl = metrics.global?.pre_fl?.[metricKey];
        if (preFl != null) {
          allValues.push(preFl as number);
        }

        metrics.rounds?.forEach((round) => {
          const value = round.validation[metricKey];
          if (value != null) {
            allValues.push(value as number);
          }
        });

        const postFl = metrics.global?.post_fl?.[metricKey];
        if (postFl != null) {
          allValues.push(postFl as number);
        }
      }
    }

    return allValues;
  };

  // Prepare data for charts (includes Pre-FL baseline as round 0)
  const prepareChartData = () => {
    if (selectedClient === "all") {
      // Show aggregate data from simulation metrics
      if (!simulationMetrics?.rounds) return [];

      const data: any[] = [];

      // Add Pre-FL baseline as round 0 (aggregate across all clients)
      if (simulationMetrics.aggregate?.pre_fl) {
        data.push({
          round: 0,
          accuracy: simulationMetrics.aggregate.pre_fl.avg_accuracy,
          precision: simulationMetrics.aggregate.pre_fl.avg_precision,
          recall: simulationMetrics.aggregate.pre_fl.avg_recall,
          f1: simulationMetrics.aggregate.pre_fl.avg_f1,
          loss: simulationMetrics.aggregate.pre_fl.avg_loss,
        });
      }

      // Add validation rounds
      simulationMetrics.rounds.forEach((round) => {
        data.push({
          round: round.round,
          accuracy: round.avg_accuracy,
          precision: round.avg_precision,
          recall: round.avg_recall,
          f1: round.avg_f1,
          loss: round.avg_loss,
        });
      });

      return data;
    } else {
      // Show individual client data
      const clientId = parseInt(selectedClient);
      const clientMetrics = clientMetricsMap.get(clientId);
      if (!clientMetrics) return [];

      const data: any[] = [];

      // Add Pre-FL baseline as round 0
      if (clientMetrics.global?.pre_fl) {
        data.push({
          round: 0,
          accuracy: clientMetrics.global.pre_fl.accuracy,
          precision: clientMetrics.global.pre_fl.precision,
          recall: clientMetrics.global.pre_fl.recall,
          f1: clientMetrics.global.pre_fl.f1_score,
          loss: clientMetrics.global.pre_fl.loss,
        });
      }

      // Add validation rounds
      clientMetrics.rounds?.forEach((round) => {
        data.push({
          round: round.round,
          accuracy: round.validation.accuracy,
          precision: round.validation.precision,
          recall: round.validation.recall,
          f1: round.validation.f1_score,
          loss: round.validation.loss,
        });
      });

      return data;
    }
  };

  // Prepare multi-client data (when "all" is selected, show individual lines)
  const prepareMultiClientData = () => {
    if (selectedClient !== "all") return null;

    // Get max rounds from any client
    let maxRounds = 0;
    clients.forEach((client) => {
      const metrics = clientMetricsMap.get(client.id);
      if (metrics?.rounds) {
        maxRounds = Math.max(maxRounds, metrics.rounds.length);
      }
    });

    // Build data with one entry per round (including round 0 for Pre-FL)
    const data: any[] = [];

    // Round 0: Pre-FL baseline
    const round0Data: any = { round: 0 };
    clients.forEach((client) => {
      const metrics = clientMetricsMap.get(client.id);
      if (metrics?.global?.pre_fl) {
        round0Data[`accuracy_${client.id}`] = metrics.global.pre_fl.accuracy;
        round0Data[`precision_${client.id}`] = metrics.global.pre_fl.precision;
        round0Data[`recall_${client.id}`] = metrics.global.pre_fl.recall;
        round0Data[`f1_${client.id}`] = metrics.global.pre_fl.f1_score;
        round0Data[`loss_${client.id}`] = metrics.global.pre_fl.loss;
      }
    });
    data.push(round0Data);

    // Validation rounds
    for (let roundNum = 1; roundNum <= maxRounds; roundNum++) {
      const roundData: any = { round: roundNum };

      clients.forEach((client) => {
        const metrics = clientMetricsMap.get(client.id);
        const roundMetrics = metrics?.rounds.find((r) => r.round === roundNum);

        if (roundMetrics) {
          roundData[`accuracy_${client.id}`] = roundMetrics.validation.accuracy;
          roundData[`precision_${client.id}`] =
            roundMetrics.validation.precision;
          roundData[`recall_${client.id}`] = roundMetrics.validation.recall;
          roundData[`f1_${client.id}`] = roundMetrics.validation.f1_score;
          roundData[`loss_${client.id}`] = roundMetrics.validation.loss;
        }
      });

      data.push(roundData);
    }

    return data;
  };

  // Prepare Pre FL vs Post FL comparison
  const prepareComparisonData = () => {
    return clients.map((client) => {
      const metrics = clientMetricsMap.get(client.id);
      return {
        name: client.client_name,
        preFl: metrics?.global?.pre_fl?.accuracy
          ? metrics.global.pre_fl.accuracy
          : 0,
        postFl: metrics?.global?.post_fl?.accuracy
          ? metrics.global.post_fl.accuracy
          : 0,
      };
    });
  };

  const chartData = prepareChartData();
  const multiClientData = prepareMultiClientData();
  const comparisonData = prepareComparisonData();

  // Collect all values for smart scaling
  const allAccuracyValues = collectAllValues("accuracy");
  const allPrecisionValues = collectAllValues("precision");
  const allRecallValues = collectAllValues("recall");
  const allF1Values = collectAllValues("f1_score");
  const allLossValues = collectAllValues("loss");

  // Calculate smart y-axis limits
  const accuracyLimits = smartYLimit(allAccuracyValues, "bounded");
  const precisionLimits = smartYLimit(allPrecisionValues, "bounded");
  const recallLimits = smartYLimit(allRecallValues, "bounded");
  const f1Limits = smartYLimit(allF1Values, "bounded");
  const lossLimits = smartYLimit(allLossValues, "unbounded");

  // Charts configuration (with smart scaling dependencies)
  const accuracyChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`accuracy_${client.id}`]) ||
                [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
            }))
          : [
              {
                label: "Accuracy",
                data: (chartData || []).map((d: any) => d.accuracy),
                borderColor: "#B80028",
                backgroundColor: "#B80028",
                tension: 0.3,
                pointRadius: 4,
              },
            ],
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [chartData, multiClientData, selectedClient, clients],
  );

  const lossChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`loss_${client.id}`]) || [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
            }))
          : [
              {
                label: "Loss",
                data: (chartData || []).map((d: any) => d.loss),
                borderColor: "#3b82f6",
                backgroundColor: "rgba(59, 130, 246, 0.1)",
                tension: 0.3,
                fill: true,
                pointRadius: 4,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const precisionChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`precision_${client.id}`]) ||
                [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
            }))
          : [
              {
                label: "Precision",
                data: (chartData || []).map((d: any) => d.precision),
                borderColor: "#10b981",
                backgroundColor: "#10b981",
                tension: 0.3,
                pointRadius: 4,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const recallChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`recall_${client.id}`]) ||
                [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
            }))
          : [
              {
                label: "Recall",
                data: (chartData || []).map((d: any) => d.recall),
                borderColor: "#f59e0b",
                backgroundColor: "#f59e0b",
                tension: 0.3,
                pointRadius: 4,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const f1ChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`f1_${client.id}`]) || [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
            }))
          : [
              {
                label: "F1 Score",
                data: (chartData || []).map((d: any) => d.f1),
                borderColor: "#8b5cf6",
                backgroundColor: "#8b5cf6",
                tension: 0.3,
                pointRadius: 4,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const comparisonChartConfig = useMemo(
    () => ({
      labels: (comparisonData || []).map((d) => d.name),
      datasets: [
        {
          label: "Pre-FL Accuracy",
          data: (comparisonData || []).map((d) => d.preFl),
          backgroundColor: "rgba(148, 163, 184, 0.6)",
          borderColor: "#94a3b8",
          borderWidth: 1,
        },
        {
          label: "Post-FL Accuracy",
          data: (comparisonData || []).map((d) => d.postFl),
          backgroundColor: "rgba(184, 0, 40, 0.7)",
          borderColor: "#B80028",
          borderWidth: 1,
        },
      ],
    }),
    [comparisonData],
  );

  // Get selected client metrics for individual view
  const selectedClientMetrics =
    selectedClient !== "all"
      ? clientMetricsMap.get(parseInt(selectedClient))
      : null;

  // Get selected client info
  const selectedClientInfo =
    selectedClient !== "all"
      ? clients.find((c) => c.id === parseInt(selectedClient))
      : null;

  // Base chart options
  const getChartOptions = (
    metricLimits: { min: number; max: number },
    isPercentage: boolean = false,
  ) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: selectedClient === "all",
        position: "top" as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          font: { size: 11 },
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#1e293b",
        bodyColor: "#1e293b",
        borderColor: "#e2e8f0",
        borderWidth: 1,
        padding: 12,
        boxPadding: 4,
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || "";
            const value = context.raw;
            if (value == null) return label;
            if (isPercentage) {
              return `${label}: ${(value * 100).toFixed(1)}%`;
            }
            return `${label}: ${value.toFixed(4)}`;
          },
          title: (context: any) => {
            const roundNum = context[0]?.label || "";
            // Add context for round 0
            if (roundNum === "Round 0") {
              return `${roundNum} (Pre-FL Baseline)`;
            }
            return roundNum;
          },
        },
      },
      annotation: {
        annotations:
          selectedClient !== "all" && selectedClientMetrics?.global?.post_fl
            ? {
                preFlLine: {
                  type: "line",
                  yMin: 0,
                  yMax: 0,
                  borderColor: "rgba(148, 163, 184, 0.5)",
                  borderWidth: 2,
                  borderDash: [5, 5],
                  label: {
                    display: true,
                    content: "Pre-FL",
                    position: "start",
                  },
                },
              }
            : {},
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        min: metricLimits.min,
        max: metricLimits.max,
        grid: {
          display: true,
          color: "rgba(0,0,0,0.05)",
        },
        ticks: {
          font: { size: 11 },
          callback: (v: any) =>
            isPercentage ? `${(v * 100).toFixed(0)}%` : v.toFixed(3),
        },
      },
      x: {
        offset: false,
        grid: {
          display: false,
        },
        ticks: {
          font: { size: 11 },
        },
      },
    },
  });

  // Metric-specific chart options with smart scaling
  const accuracyChartOptions = getChartOptions(accuracyLimits, true);
  const precisionChartOptions = getChartOptions(precisionLimits, true);
  const recallChartOptions = getChartOptions(recallLimits, true);
  const f1ChartOptions = getChartOptions(f1Limits, true);
  const lossChartOptions = getChartOptions(lossLimits, false);

  // if (isLoading) {
  //   return (
  //     <div className="p-8">
  //       <div className="text-center text-muted-foreground">
  //         Loading simulation data...
  //       </div>
  //     </div>
  //   );
  // }

  // if (!simulation) {
  //   return (
  //     <div className="p-8">
  //       <div className="text-center text-muted-foreground">
  //         Simulation not found
  //       </div>
  //     </div>
  //   );
  // }

  // Check if metrics are empty (not needed with static data)
  // const hasMetrics =
  //   simulationMetrics &&
  //   simulationMetrics.rounds &&
  //   simulationMetrics.rounds.length > 0;

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-12">
        <button
          onClick={onBack}
          className="group flex items-center gap-2 text-slate-500 hover:text-slate-900 mb-6 transition-colors"
        >
          <div className="p-2 rounded-full bg-slate-100 group-hover:bg-slate-200 transition-colors">
            <ArrowLeft size={16} />
          </div>
          <span className="font-medium text-sm">Back to History</span>
        </button>
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-4 mb-3">
              <h1 className="text-4xl font-bold text-slate-900 tracking-tight">
                {simulationName}
              </h1>
              {selectedClient !== "all" && (
                <span className="px-3 py-1 bg-blue-50 text-blue-600 border border-blue-100 text-xs font-semibold rounded-full uppercase tracking-wide">
                  Individual View
                </span>
              )}
              {selectedClient === "all" && (
                <span className="px-3 py-1 bg-purple-50 text-purple-600 border border-purple-100 text-xs font-semibold rounded-full uppercase tracking-wide">
                  Multi-Client View
                </span>
              )}
            </div>
            <p className="text-slate-500 text-lg">
              Simulation ID:{" "}
              <span className="font-mono text-slate-700 bg-slate-100 px-1 py-0.5 rounded text-base">
                {simulationId}
              </span>{" "}
              •{" "}
              <span className="font-medium text-slate-700">
                {clients.length}
              </span>{" "}
              Clients •{" "}
              <span
                className={`font-medium ${simulation?.status === "completed" ? "text-green-600" : "text-slate-700"}`}
              >
                {simulation?.status
                  ? simulation.status.charAt(0).toUpperCase() +
                    simulation.status.slice(1)
                  : "Unknown"}
              </span>
            </p>
          </div>

          {/* Client Selector */}
          <div className="flex items-center gap-3">
            <Users size={18} className="text-muted-foreground" />
            <Select value={selectedClient} onValueChange={setSelectedClient}>
              <SelectTrigger className="w-[280px] h-10 font-medium">
                <SelectValue placeholder="Select client" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all" className="font-medium">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500"></div>
                    <span>All Clients (Multi-Line View)</span>
                  </div>
                </SelectItem>
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Individual Clients
                </div>
                {clients.map((client, idx) => (
                  <SelectItem
                    key={client.id}
                    value={client.id.toString()}
                    className="pl-6"
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{
                          backgroundColor:
                            CLIENT_COLORS[idx % CLIENT_COLORS.length],
                        }}
                      ></div>
                      <span>
                        {client.client_name}{" "}
                        <span className="text-xs text-muted-foreground">
                          ({client.model_type})
                        </span>
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Individual Client Summary (when specific client selected) */}
      {selectedClient !== "all" && selectedClientMetrics && (
        <div className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
              Client Performance Summary
            </h2>
            <div className="px-4 py-1.5 bg-slate-100 rounded-full text-sm font-medium text-slate-600">
              {selectedClientInfo?.client_name}{" "}
              <span className="text-slate-400">|</span>{" "}
              <span className="text-slate-500">
                {selectedClientInfo?.model_type}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {/* Pre-FL Accuracy */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Pre-FL Baseline
              </div>
              <div className="text-3xl font-bold text-slate-700">
                {(
                  (selectedClientMetrics.global?.pre_fl?.accuracy || 0) * 100
                ).toFixed(1)}
                <span className="text-lg text-slate-400 ml-1">%</span>
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                Initial Accuracy
              </div>
            </div>

            {/* Post-FL Accuracy */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow ring-1 ring-green-100">
              <div className="text-xs font-semibold text-green-600 uppercase tracking-wider mb-2">
                Post-FL Result
              </div>
              <div className="text-3xl font-bold text-slate-900">
                {(
                  (selectedClientMetrics.global?.post_fl?.accuracy || 0) * 100
                ).toFixed(1)}
                <span className="text-lg text-slate-400 ml-1">%</span>
              </div>
              <div className="text-sm text-green-600 mt-2 font-medium">
                Final Accuracy
              </div>
            </div>

            {/* Improvement */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Net Improvement
              </div>
              <div
                className={`text-3xl font-bold ${
                  (selectedClientMetrics.global?.improvement?.accuracy || 0) >=
                  0
                    ? "text-blue-600"
                    : "text-red-500"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.accuracy || 0) >= 0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.accuracy || 0) *
                  100
                ).toFixed(1)}
                <span className="text-lg text-slate-400 ml-1">%</span>
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                Accuracy Gain
              </div>
            </div>

            {/* Best Round */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Peak Performance
              </div>
              <div className="text-3xl font-bold text-purple-600">
                Round{" "}
                {selectedClientMetrics.rounds?.reduce(
                  (best, round) =>
                    round.validation.accuracy > (best?.validation.accuracy || 0)
                      ? round
                      : best,
                  selectedClientMetrics.rounds[0],
                )?.round || "N/A"}
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                {(
                  (selectedClientMetrics.rounds?.reduce(
                    (best, round) =>
                      round.validation.accuracy >
                      (best?.validation.accuracy || 0)
                        ? round
                        : best,
                    selectedClientMetrics.rounds[0],
                  )?.validation.accuracy || 0) * 100
                ).toFixed(1)}
                % Accuracy
              </div>
            </div>

            {/* Total Rounds */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Duration
              </div>
              <div className="text-3xl font-bold text-slate-700">
                {selectedClientMetrics.rounds?.length || 0}
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                Rounds Completed
              </div>
            </div>
          </div>

          {/* Additional Metrics Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-6">
            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  F1 Score
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(
                    selectedClientMetrics.global?.post_fl?.f1_score || 0
                  ).toFixed(3)}
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.f1_score || 0) >=
                  0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.f1_score || 0) >= 0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.f1_score || 0) *
                  100
                ).toFixed(1)}
                %
              </div>
            </div>

            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  Precision
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(
                    (selectedClientMetrics.global?.post_fl?.precision || 0) *
                    100
                  ).toFixed(1)}
                  %
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.precision || 0) >=
                  0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.precision || 0) >=
                0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.precision || 0) *
                  100
                ).toFixed(1)}
                %
              </div>
            </div>

            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  Recall
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(
                    (selectedClientMetrics.global?.post_fl?.recall || 0) * 100
                  ).toFixed(1)}
                  %
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.recall || 0) >= 0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.recall || 0) >= 0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.recall || 0) * 100
                ).toFixed(1)}
                %
              </div>
            </div>

            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  Loss
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(selectedClientMetrics.global?.post_fl?.loss || 0).toFixed(
                    4,
                  )}
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.loss || 0) <= 0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.loss || 0).toFixed(
                  4,
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* View Mode Indicator */}
      <div className="mb-4 flex items-center gap-2 text-sm text-slate-500">
        {selectedClient === "all" ? (
          <>
            <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500"></div>
            <span>
              Viewing{" "}
              <strong className="text-slate-900 font-semibold">
                all clients
              </strong>{" "}
              with individual trend lines per client
            </span>
          </>
        ) : (
          <>
            <div
              className="w-2 h-2 rounded-full"
              style={{
                backgroundColor:
                  CLIENT_COLORS[
                    clients.findIndex(
                      (c) => c.id === parseInt(selectedClient),
                    ) % CLIENT_COLORS.length
                  ],
              }}
            ></div>
            <span>
              Viewing{" "}
              <strong className="text-slate-900 font-semibold">
                {selectedClientInfo?.client_name}
              </strong>{" "}
              individual performance
              {" • "}Round 0 = Pre-FL baseline
            </span>
          </>
        )}
      </div>

      {/* Statistics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8 mt-6">
        {/* Accuracy Progression Chart */}
        <EvalCard
          title="Accuracy"
          description={
            selectedClient === "all"
              ? "Multi-client accuracy progression (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} accuracy over time`
          }
          content={
            <Line data={accuracyChartConfig} options={accuracyChartOptions} />
          }
        />

        {/* Loss Progression Chart */}
        <EvalCard
          title="Loss"
          description={
            selectedClient === "all"
              ? "Multi-client loss convergence (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} loss convergence`
          }
          content={<Line data={lossChartConfig} options={lossChartOptions} />}
        />

        {/* Precision Chart */}
        <EvalCard
          title="Precision"
          description={
            selectedClient === "all"
              ? "Multi-client precision trends (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} precision progression`
          }
          content={
            <Line data={precisionChartConfig} options={precisionChartOptions} />
          }
        />

        {/* Recall Chart */}
        <EvalCard
          title="Recall"
          description={
            selectedClient === "all"
              ? "Multi-client recall/sensitivity (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} recall progression`
          }
          content={
            <Line data={recallChartConfig} options={recallChartOptions} />
          }
        />

        {/* F1 Score Chart */}
        <EvalCard
          title="F1 Score"
          description={
            selectedClient === "all"
              ? "Multi-client F1 score trends (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} F1 score progression`
          }
          content={<Line data={f1ChartConfig} options={f1ChartOptions} />}
        />

        {/* Pre FL vs Post FL Comparison */}
        <EvalCard
          title="FL Impact"
          description={
            selectedClient === "all"
              ? "Pre-FL vs Post-FL comparison across all clients"
              : "Pre-FL baseline vs Post-FL final performance"
          }
          content={
            <Bar data={comparisonChartConfig} options={accuracyChartOptions} />
          }
        />
      </div>
    </div>
  );
}
