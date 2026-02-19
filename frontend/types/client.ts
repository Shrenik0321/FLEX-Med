// Client types matching backend schema

export interface Client {
  id: number;
  client_name: string;
  status: "Active" | "Inactive";
  model_type: string;
  model_path: string;
  created_at?: string;
  client_email?: string;
}

export interface ClientCreate {
  client_name: string;
  status: "Active" | "Inactive";
  model_type: string;
  model_path?: string;
}
