"use client";

import { Search, Trash2, MoreVertical, Play } from "lucide-react";
import { useEffect, useState, useRef } from "react";
import { toast } from "sonner";

// Helper to format ISO date strings to dd/mm/yy
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

// Confirmation Modal Component
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
    <div className="fixed inset-0 bg-gray-900 bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-card rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
        <h2 className="text-xl font-semibold text-foreground mb-4">
          Confirm Deletion
        </h2>
        <p className="text-gray-700 mb-6">
          Are you sure you want to delete{" "}
          <span className="font-semibold">{clientName}</span>? This action
          cannot be undone.
        </p>
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            disabled={isDeleting}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isDeleting}
            className="px-4 py-2 bg-primary text-white rounded-lg font-medium hover:bg-[#9a0022] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}

// Action Dropdown Component
interface ActionDropdownProps {
  client: Client;
  onDelete: (client: Client) => void;
  onStartTraining: (client: Client) => void;
}

function ActionDropdown({
  client,
  onDelete,
  onStartTraining,
}: ActionDropdownProps) {
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

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="p-1 hover:bg-gray-100 rounded transition-colors"
        title="Actions"
      >
        <MoreVertical size={18} className="text-gray-600" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-card rounded-lg shadow-lg border border-border py-1 z-50">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onStartTraining(client);
              setIsOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors"
          >
            <Play size={16} className="text-green-600" />
            Start Training
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(client);
              setIsOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-sm text-primary hover:bg-red-50 flex items-center gap-2 transition-colors"
          >
            <Trash2 size={16} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

export default function ClientsListPage({
  onAddClick,
  onSelectClient,
}: ClientsListPageProps) {
  const [clients, setClients] = useState<Client[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [deleteModal, setDeleteModal] = useState<{
    isOpen: boolean;
    clientId: number | null;
    clientName: string;
  }>({
    isOpen: false,
    clientId: null,
    clientName: "",
  });
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const fetchClients = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/clients");
        if (!response.ok) throw new Error("Failed to fetch");
        const data = await response.json();
        setClients(data);
      } catch (error) {
        console.error("Failed to fetch clients:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchClients();
  }, []);

  const handleDeleteClick = (client: Client) => {
    setDeleteModal({
      isOpen: true,
      clientId: client.id,
      clientName: client.client_name,
    });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteModal.clientId) return;

    setIsDeleting(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/clients/${deleteModal.clientId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) throw new Error("Failed to delete");

      // Remove the deleted client from the state
      setClients((prevClients) =>
        prevClients.filter((client) => client.id !== deleteModal.clientId)
      );

      // Close the modal
      setDeleteModal({ isOpen: false, clientId: null, clientName: "" });
    } catch (error) {
      console.error("Failed to delete client:", error);
      toast.error("Failed to delete client. Please try again.");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteModal({ isOpen: false, clientId: null, clientName: "" });
  };

  const handleStartTraining = async (client: Client) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/clients/${client.id}/start_local_train`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) throw new Error("Failed to start training");

      const data = await response.json();
      console.log("Training started:", data);

      // Optionally update the client's status to "Training"
      setClients((prevClients) =>
        prevClients.map((c) =>
          c.id === client.id ? { ...c, status: "Training" } : c
        )
      );

      toast.success(`Training started for ${client.client_name}`);
    } catch (error) {
      console.error("Failed to start training:", error);
      toast.error("Failed to start training. Please try again.");
    }
  };

  if (loading) {
    return <div className="p-8">Loading clients...</div>;
  }

  const filteredClients = clients.filter((client) => {
    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    return (
      client.client_name.toLowerCase().includes(term) ||
      client.client_email.toLowerCase().includes(term) ||
      client.status.toLowerCase().includes(term)
    );
  });

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">Clients</h1>
          <p className="text-muted-foreground mt-2">
            Manage all your healthcare clients and their ML deployments
          </p>
        </div>
        <button
          onClick={onAddClick}
          className="px-4 py-2 bg-card border border-primary text-primary rounded-lg font-medium hover:bg-primary/10 transition-colors"
        >
          Add New Client
        </button>
      </div>

      {/* Search */}
      <div className="mb-6">
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
            size={18}
          />
          <input
            type="text"
            placeholder="Search clients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
          />
        </div>
      </div>

      {/* Clients Table */}
      <div className="bg-card rounded-lg shadow-sm border border-border">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                Name
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                Models
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                Created At
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredClients.map((client) => (
              <tr
                key={client.id}
                onClick={() => onSelectClient(client)}
                className="cursor-pointer border-b border-border hover:bg-[rgba(184,0,40,0.03)] transition-colors"
              >
                <td className="px-6 py-4 text-sm font-medium text-foreground">
                  {client.client_name}
                </td>
                <td className="px-6 py-4 text-sm">
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-xs font-medium border ${
                      client.status === "Training"
                        ? "border-primary text-primary bg-[rgba(184,0,40,0.05)]"
                        : client.status === "Completed"
                        ? "border-green-500 text-green-700 bg-green-50"
                        : "border-[#718096] text-muted-foreground bg-gray-50"
                    }`}
                  >
                    {client.status}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-foreground">
                  {client.model_type}
                </td>
                <td className="px-6 py-4 text-sm text-foreground">
                  {formatDate(client.created_at)}
                </td>
                <td className="px-6 py-4 text-sm">
                  <ActionDropdown
                    client={client}
                    onDelete={handleDeleteClick}
                    onStartTraining={handleStartTraining}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={deleteModal.isOpen}
        clientName={deleteModal.clientName}
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
        isDeleting={isDeleting}
      />
    </div>
  );
}
