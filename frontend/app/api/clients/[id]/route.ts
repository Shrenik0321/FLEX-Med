import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

export async function PUT(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const supabase = await createClient();
    const { id } = params;
    // Convert id to a number to match the bigint column type in the database
    const clientId = Number(id);
    console.log("Client ID:", clientId);
    if (Number.isNaN(clientId)) {
      console.error("Invalid client id provided:", id);
      return NextResponse.json({ error: "Invalid client id" }, { status: 400 });
    }

    const body = await request.json();
    const { client_name, client_email, status, model_type } = body;

    const { data, error } = await supabase
      .from("clients")
      .update({ client_name, client_email, status, model_type })
      .eq("id", clientId)
      .select();

    if (error) {
      console.error("Error updating client:", error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to update client:", error);
    return NextResponse.json(
      { error: "Failed to update client" },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const supabase = await createClient();
    const { id } = params;
    // Convert id to a number to match the bigint column type in the database
    const clientId = Number(id);
    console.log("Client ID:", clientId);
    if (Number.isNaN(clientId)) {
      console.error("Invalid client id provided:", id);
      return NextResponse.json({ error: "Invalid client id" }, { status: 400 });
    }

    const { error } = await supabase
      .from("clients")
      .delete()
      .eq("id", clientId);

    if (error) {
      console.error("Error deleting client:", error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete client:", error);
    return NextResponse.json(
      { error: "Failed to delete client" },
      { status: 500 }
    );
  }
}
