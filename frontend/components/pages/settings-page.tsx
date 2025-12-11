"use client"

export default function SettingsPage() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-semibold text-gray-900 mb-2">Settings</h1>
      <p className="text-[#718096] mb-8">Manage your dashboard preferences and system configuration</p>

      <div className="max-w-2xl space-y-6">
        {/* Notification Settings */}
        <div className="bg-white rounded-lg p-6 flex-card-shadow">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            Notifications
          </h2>
          <div className="space-y-3">
            {[
              { label: "Email alerts on training completion", enabled: true },
              { label: "Push notifications for anomalies", enabled: true },
              { label: "Weekly summary reports", enabled: false },
            ].map((setting, idx) => (
              <div key={idx} className="flex items-center justify-between">
                <span className="text-sm text-gray-900">{setting.label}</span>
                <button
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    setting.enabled ? "bg-[#B80028]" : "bg-[#E2E8F0]"
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      setting.enabled ? "translate-x-6" : "translate-x-1"
                    }`}
                  />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Display Settings */}
        <div className="bg-white rounded-lg p-6 flex-card-shadow">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            Display
          </h2>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-900 block mb-2">Theme</label>
              <select className="w-full px-4 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent">
                <option>Light</option>
                <option>Dark</option>
                <option>System</option>
              </select>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex gap-3">
          <button className="px-6 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}
