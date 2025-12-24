"use client";

import AddClientPage from "@/components/pages/add-client-page";
import { useRouter } from "next/navigation";

export default function AddClient() {
  const router = useRouter();

  return <AddClientPage onBack={() => router.push("/clients")} />;
}
