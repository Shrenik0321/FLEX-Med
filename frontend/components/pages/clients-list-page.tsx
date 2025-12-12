"use client";

import { Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

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
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
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
            className="px-4 py-2 bg-[#B80028] text-white rounded-lg font-medium hover:bg-[#9a0022] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
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
      const response = await fetch(`http://localhost:8000/api/clients/${deleteModal.clientId}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Failed to delete");

      // Remove the deleted client from the state
      setClients((prevClients) =>
        prevClients.filter((client) => client.id !== deleteModal.clientId)
      );

      // Close the modal
      setDeleteModal({ isOpen: false, clientId: null, clientName: "" });
    } catch (error) {
      console.error("Failed to delete client:", error);
      alert("Failed to delete client. Please try again.");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteModal({ isOpen: false, clientId: null, clientName: "" });
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
          <h1 className="text-3xl font-semibold text-gray-900">Clients</h1>
          <p className="text-[#718096] mt-2">
            Manage all your healthcare clients and their ML deployments
          </p>
        </div>
        <button
          onClick={onAddClick}
          className="px-4 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors"
        >
          Add New Client
        </button>
      </div>

      {/* Search */}
      <div className="mb-6">
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#718096]"
            size={18}
          />
          <input
            type="text"
            placeholder="Search clients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
          />
        </div>
      </div>

      {/* Clients Table */}
      <div className="bg-white rounded-lg overflow-hidden flex-card-shadow">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[#E2E8F0]">
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider">
                Name
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider">
                Email
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider">
                Models
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider">
                Created At
              </th>
              <th className="px-6 py-4 text-left text-xs font-semibold text-gray-900 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredClients.map((client) => (
              <tr
                key={client.id}
                onClick={() => onSelectClient(client)}
                className="cursor-pointer border-b border-[#E2E8F0] hover:bg-[rgba(184,0,40,0.03)] transition-colors"
              >
                <td className="px-6 py-4 text-sm font-medium text-gray-900">
                  {client.client_name}
                </td>
                <td className="px-6 py-4 text-sm text-[#718096]">
                  {client.client_email}
                </td>
                <td className="px-6 py-4 text-sm">
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-xs font-medium border ${
                      client.status === "Training"
                        ? "border-[#B80028] text-[#B80028] bg-[rgba(184,0,40,0.05)]"
                        : client.status === "Completed"
                        ? "border-green-500 text-green-700 bg-green-50"
                        : "border-[#718096] text-[#718096] bg-gray-50"
                    }`}
                  >
                    {client.status}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-900">
                  {client.model_type}
                </td>
                <td className="px-6 py-4 text-sm text-gray-900">
                  {formatDate(client.created_at)}
                </td>
                <td className="px-6 py-4 text-sm">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteClick(client);
                    }}
                    className="text-[#B80028] hover:text-[#9a0022] transition-colors"
                    title="Delete client"
                  >
                    <Trash2 size={18} />
                  </button>
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
