import { Outlet } from 'react-router-dom';
import Sidebar from '../components/Sidebar';

// Two-pane shell: persistent search/compound sidebar on the left, the selected
// compound's report (or an empty state) on the right via <Outlet>.
export default function AppLayout() {
  return (
    <div className="layout">
      <Sidebar />
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
