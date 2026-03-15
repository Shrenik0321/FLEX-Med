"use client";

import { API_BASE_PATH } from "@/utils";
import { Trash2, MoreVertical, Loader2 } from "lucide-react";
import { useEffect, useState, useRef } from "react";
import { toast } from "sonner";
import { Loading } from "@/components/ui/loading";
import { EmptyState } from "@/components/ui/empty-state";

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const day = String(date.getDate()).padStart(2, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const year = String(date.getFullYear()).slice(-2);
  return `${day}/${month}/${year}`;
}

interface ClientsListPageProps {
  onAddClick: () => void;
  onSelectClient: (client: Client) => void;
}

interface Client {
  id: number;
  client_name: string;
  client_email: string;
  status: string;
  model_type: number;
  created_at: string;
}

interface DeleteConfirmationModalProps {
  isOpen: boolean;
  clientName: string;
  onConfirm: () => void;
  onCancel: () => void;
  isDeleting: boolean;
}

function DeleteConfirmationModal({
  isOpen,
  clientName,
  onConfirm,
  onCancel,
  isDeleting,
}: DeleteConfirmationModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-900/20 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl border border-gray-100 p-6 max-w-sm w-full mx-4 shadow-xl">
        <p className="text-base font-semibold text-gray-900 mb-2">
          Delete client?
        </p>
        <p className="text-sm text-gray-500 mb-6">
          <span className="font-medium text-gray-700">{clientName}</span> will
          be permanently removed. This cannot be undone.
        </p>
        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            disabled={isDeleting}
            className="px-3 py-1.5 rounded-lg text-sm font-medium text-gray-500 hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isDeleting}
            className="px-3 py-1.5 rounded-lg text-sm font-medium bg-red-600 text-white hover:bg-red-700 transition-colors inline-flex items-center gap-1.5"
          >
            {isDeleting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Deleting…
              </>
            ) : (
              "Delete"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

interface ActionDropdownProps {
  client: Client;
  onDelete: (client: Client) => void;
}

function ActionDropdown({ client, onDelete }: ActionDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    if (isOpen) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="p-1.5 rounded-md hover:bg-gray-100 transition-colors"
        title="Actions"
      >
        <MoreVertical className="h-3.5 w-3.5 text-gray-400" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-1 w-40 bg-white rounded-lg border border-gray-100 shadow-lg py-1 z-50">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(client);
              setIsOpen(false);
            }}
            className="w-full px-3 py-2 text-left text-xs text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
          >
            <Trash2 size={13} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

function StatusTag({ status }: { status: string }) {
  const cls =
    status === "Active"
      ? "text-green-700 bg-green-50 border-green-300"
      : status === "Inactive"
        ? "text-red-700 bg-red-50 border-red-300"
        : status === "Training"
          ? "text-blue-700 bg-blue-50 border-blue-200"
          : "text-gray-500 bg-gray-50 border-gray-200";
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold border ${cls}`}
    >
      {status}
    </span>
  );
}

export default function ClientsListPage({
  onAddClick,
  onSelectClient,
}: ClientsListPageProps) {
  const [clients, setClients] = useState<Client[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deleteModal, setDeleteModal] = useState<{
    isOpen: boolean;
    clientId: number | null;
    clientName: string;
  }>({ isOpen: false, clientId: null, clientName: "" });
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const fetchClients = async () => {
      try {
        const response = await fetch(`${API_BASE_PATH}/clients`);
        if (!response.ok) throw new Error("Failed to fetch");
        const data = await response.json();
        setClients(data);
      } catch (error) {
        console.error("Failed to fetch clients:", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchClients();
  }, []);

  const handleDeleteClick = (client: Client) => {
    setDeleteModal({ isOpen: true, clientId: client.id, clientName: client.client_name });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteModal.clientId) return;
    setIsDeleting(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/clients/${deleteModal.clientId}`,
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error("Failed to delete");
      setClients((prev) => prev.filter((c) => c.id !== deleteModal.clientId));
      setDeleteModal({ isOpen: false, clientId: null, clientName: "" });
    } catch (error) {
      console.error("Failed to delete client:", error);
      toast.error("Failed to delete client. Please try again.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="px-6 py-8">
      {isLoading ? (
        <Loading className="min-h-[400px]" text="Loading clients…" />
      ) : (
        <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
          {clients.length === 0 ? (
            <EmptyState
              title="No clients found"
              description="There are no clients connected to the network."
              action={{ label: "Add Client", onClick: onAddClick }}
            />
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100">
                  {["Name", "Status", "Model", "Created", ""].map((h) => (
                    <th
                      key={h}
                      className={`py-3 px-4 font-semibold uppercase tracking-wider text-xs text-gray-400 ${
                        h === "Name" ? "text-left" : h === "" ? "text-right" : "text-left"
                      }`}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {clients.map((client) => (
                  <tr
                    key={client.id}
                    onClick={() => onSelectClient(client)}
                    className="cursor-pointer border-b border-gray-50 last:border-0 hover:bg-gray-50/60 transition-colors"
                  >
                    <td className="py-3.5 px-4 font-semibold text-gray-900">
                      {client.client_name}
                    </td>
                    <td className="py-3.5 px-4">
                      <StatusTag status={client.status} />
                    </td>
                    <td className="py-3.5 px-4 text-gray-500">
                      {client.model_type}
                    </td>
                    <td className="py-3.5 px-4 text-gray-400 tabular-nums">
                      {formatDate(client.created_at)}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <ActionDropdown
                        client={client}
                        onDelete={handleDeleteClick}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <DeleteConfirmationModal
        isOpen={deleteModal.isOpen}
        clientName={deleteModal.clientName}
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteModal({ isOpen: false, clientId: null, clientName: "" })}
        isDeleting={isDeleting}
      />

      {isDeleting && <Loading fullScreen text="Removing client from network…" />}
    </div>
  );
}
