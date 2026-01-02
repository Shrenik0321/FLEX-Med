"use client";

export default function HomePage() {
  const stats = [
    { label: "Total Clients", value: "24", icon: "👥" },
    { label: "Active Training", value: "8", icon: "🔄", indicator: true },
    { label: "Models Deployed", value: "12", icon: "✓", color: "green" },
    { label: "Alerts", value: "3", icon: "⚠️", color: "red" },
  ];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-foreground flex items-center gap-3">
          Dashboard Overview
          <span className="flex-red-underline inline-block w-12" />
        </h1>
        <p className="text-muted-foreground mt-2">
          Welcome back! Here's your healthcare ML infrastructure at a glance.
        </p>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, idx) => (
          <div
            key={idx}
            className="bg-card rounded-lg p-6 shadow-sm border border-border"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-muted-foreground text-sm font-medium flex items-center gap-2">
                  <span className="flex-red-dot" />
                  {stat.label}
                </p>
                <p className="text-3xl font-bold text-foreground mt-2">
                  {stat.value}
                </p>
              </div>
              <span className="text-2xl">{stat.icon}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4">
            Training Progress
          </h2>
          <div className="space-y-4">
            {[
              { name: "Model A", progress: 85, status: "Training" },
              { name: "Model B", progress: 60, status: "Training" },
              { name: "Model C", progress: 100, status: "Completed" },
            ].map((item, idx) => (
              <div key={idx}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-foreground">
                    {item.name}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {item.progress}%
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${item.progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex-red-underline">
            Recent Activity
          </h2>
          <div className="space-y-3">
            {[
              {
                action: "Model training completed",
                time: "2 hours ago",
                type: "success",
              },
              {
                action: "New client registered",
                time: "5 hours ago",
                type: "info",
              },
              {
                action: "Alert: High inference latency",
                time: "1 day ago",
                type: "alert",
              },
            ].map((item, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3 pb-3 border-b border-border last:border-0"
              >
                <div
                  className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                    item.type === "success"
                      ? "bg-green-500"
                      : item.type === "alert"
                      ? "bg-primary"
                      : "bg-blue-500"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground">
                    {item.action}
                  </p>
                  <p className="text-xs text-muted-foreground">{item.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
