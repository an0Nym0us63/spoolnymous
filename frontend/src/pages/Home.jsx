import React, { useEffect } from "react";
import { usePrinter } from "../store/printer";
import PrinterCard from "../components/PrinterCard";
import AMSGrid from "../components/AMSGrid";
import StatusBanner from "../components/StatusBanner";

export default function Home() {
  const { status, startPolling, stopPolling } = usePrinter();

  useEffect(() => {
    startPolling(3000);
    return () => stopPolling();
  }, []);

  return (
    <div className="space-y-4">
      <StatusBanner status={status} />
      <PrinterCard status={status} />
      <AMSGrid amsList={status?.ams_list ?? []} activeTray={null} />
    </div>
  );
}
