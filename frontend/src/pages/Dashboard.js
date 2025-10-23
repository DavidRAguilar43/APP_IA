import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp, DollarSign, ShoppingCart, Package } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a'];

const Dashboard = () => {
  const [salesData, setSalesData] = useState([]);
  const [stats, setStats] = useState(null);
  const [chartData, setChartData] = useState({});
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [categories, setCategories] = useState([]);
  
  // Table controls
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("");
  const [sortOrder, setSortOrder] = useState("asc");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [totalRecords, setTotalRecords] = useState(0);

  // Cargar categorías
  useEffect(() => {
    fetchCategories();
  }, []);

  // Cargar datos iniciales
  useEffect(() => {
    fetchAllData();
  }, [page, search, sortBy, sortOrder, categoryFilter]);

  const fetchCategories = async () => {
    try {
      const response = await axios.get(`${API}/sales/categories`);
      setCategories(response.data.categories);
    } catch (error) {
      console.error("Error cargando categorías:", error);
    }
  };

  const fetchAllData = async () => {
    setLoading(true);
    try {
      // Cargar datos de la tabla
      const salesResponse = await axios.get(`${API}/sales/data`, {
        params: {
          page,
          page_size: pageSize,
          search: search || undefined,
          sort_by: sortBy || undefined,
          sort_order: sortOrder,
          category_filter: categoryFilter
        }
      });
      setSalesData(salesResponse.data.data);
      setTotalRecords(salesResponse.data.total);

      // Cargar estadísticas
      const statsResponse = await axios.get(`${API}/sales/stats`);
      setStats(statsResponse.data.stats);

      // Cargar datos de gráficas
      const chartTypes = [
        'sales_by_category',
        'sales_trend',
        'price_distribution',
        'top_products',
        'quantity_vs_total',
        'quantity_by_category',
        'profit_by_category'
      ];

      const chartPromises = chartTypes.map(type =>
        axios.get(`${API}/sales/charts`, { params: { chart_type: type } })
      );

      const chartResponses = await Promise.all(chartPromises);
      const chartsObj = {};
      chartTypes.forEach((type, index) => {
        chartsObj[type] = chartResponses[index].data.chart_data;
      });
      setChartData(chartsObj);

      // Cargar predicciones
      const predResponse = await axios.get(`${API}/sales/prediction`);
      setPredictions(predResponse.data);

      toast.success("Datos cargados exitosamente");
    } catch (error) {
      console.error("Error cargando datos:", error);
      toast.error("Error cargando datos del dashboard");
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(column);
      setSortOrder("asc");
    }
    setPage(1);
  };

  const handleSearch = (value) => {
    setSearch(value);
    setPage(1);
  };

  const handleCategoryFilter = (value) => {
    setCategoryFilter(value);
    setPage(1);
  };

  if (loading) {
    return <div className="loading">Cargando dashboard...</div>;
  }

  if (!stats) {
    return <div className="error">Error al cargar los datos</div>;
  }

  const totalPages = Math.ceil(totalRecords / pageSize);

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <h1 data-testid="dashboard-title">📊 Dashboard de Análisis de Ventas</h1>
        <p>Análisis completo con estadísticas descriptivas y predictivas</p>
      </div>

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card" data-testid="total-sales-card">
          <h3><DollarSign size={16} style={{display: 'inline'}} /> Ventas Totales</h3>
          <p>${stats.ventas_totales.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</p>
        </div>
        <div className="stat-card" data-testid="total-profit-card">
          <h3><TrendingUp size={16} style={{display: 'inline'}} /> Ganancia Total</h3>
          <p>${stats.ganancia_total.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</p>
        </div>
        <div className="stat-card" data-testid="num-sales-card">
          <h3><ShoppingCart size={16} style={{display: 'inline'}} /> Número de Ventas</h3>
          <p>{stats.numero_ventas}</p>
        </div>
        <div className="stat-card" data-testid="avg-ticket-card">
          <h3><Package size={16} style={{display: 'inline'}} /> Ticket Promedio</h3>
          <p>${stats.ticket_promedio.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</p>
        </div>
      </div>

      {/* Tabla Interactiva */}
      <div className="section-card">
        <h2 className="section-title" data-testid="interactive-table-title">📋 Tabla Interactiva de Ventas</h2>
        
        <div className="controls-bar">
          <input
            type="text"
            className="search-input"
            placeholder="Buscar..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            data-testid="search-input"
          />
          <select
            className="select-input"
            value={categoryFilter}
            onChange={(e) => handleCategoryFilter(e.target.value)}
            data-testid="category-filter"
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>
                {cat === 'all' ? 'Todas las categorías' : cat}
              </option>
            ))}
          </select>
        </div>

        <div className="table-container">
          <table className="sales-table" data-testid="sales-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('Venta_ID')} data-testid="sort-venta-id">Venta ID {sortBy === 'Venta_ID' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                <th onClick={() => handleSort('Producto')} data-testid="sort-producto">Producto {sortBy === 'Producto' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                <th onClick={() => handleSort('Categoría')} data-testid="sort-categoria">Categoría {sortBy === 'Categoría' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                <th onClick={() => handleSort('Cantidad')} data-testid="sort-cantidad">Cantidad {sortBy === 'Cantidad' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                <th onClick={() => handleSort('Precio_Unitario')} data-testid="sort-precio-unitario">Precio Unit. {sortBy === 'Precio_Unitario' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                <th onClick={() => handleSort('Precio_Total')} data-testid="sort-precio-total">Precio Total {sortBy === 'Precio_Total' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                <th onClick={() => handleSort('Fecha')} data-testid="sort-fecha">Fecha {sortBy === 'Fecha' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
              </tr>
            </thead>
            <tbody>
              {salesData.map((row, index) => (
                <tr key={index} data-testid={`table-row-${index}`}>
                  <td>{row.Venta_ID}</td>
                  <td>{row.Producto}</td>
                  <td>{row.Categoría}</td>
                  <td>{row.Cantidad}</td>
                  <td>${row.Precio_Unitario.toFixed(2)}</td>
                  <td>${row.Precio_Total.toFixed(2)}</td>
                  <td>{row.Fecha}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="pagination">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            data-testid="prev-page-btn"
          >
            Anterior
          </button>
          <span data-testid="page-info">Página {page} de {totalPages}</span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            data-testid="next-page-btn"
          >
            Siguiente
          </button>
        </div>
      </div>

      {/* Gráficas */}
      <div className="section-card">
        <h2 className="section-title" data-testid="charts-title">📈 Visualizaciones Interactivas</h2>
        <p style={{marginBottom: '2rem', color: '#666'}}>Explora diferentes perspectivas de tus datos de ventas</p>
        
        <div className="charts-grid">
          {/* Gráfica 1: Ventas por Categoría */}
          {chartData.sales_by_category && (
            <div className="chart-card" data-testid="chart-sales-by-category">
              <h3 className="chart-title">{chartData.sales_by_category.title}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData.sales_by_category.labels.map((label, i) => ({
                  name: label,
                  value: chartData.sales_by_category.values[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
                  <Bar dataKey="value" fill={COLORS[0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfica 2: Tendencia de Ventas */}
          {chartData.sales_trend && (
            <div className="chart-card" data-testid="chart-sales-trend">
              <h3 className="chart-title">{chartData.sales_trend.title}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData.sales_trend.labels.map((label, i) => ({
                  date: label,
                  value: chartData.sales_trend.values[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={false} />
                  <YAxis />
                  <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
                  <Line type="monotone" dataKey="value" stroke={COLORS[1]} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfica 3: Distribución de Precios */}
          {chartData.price_distribution && (
            <div className="chart-card" data-testid="chart-price-distribution">
              <h3 className="chart-title">{chartData.price_distribution.title}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData.price_distribution.labels.map((label, i) => ({
                  range: label,
                  count: chartData.price_distribution.values[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill={COLORS[2]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfica 4: Top Productos */}
          {chartData.top_products && (
            <div className="chart-card" data-testid="chart-top-products">
              <h3 className="chart-title">{chartData.top_products.title}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData.top_products.labels.map((label, i) => ({
                  name: label,
                  value: chartData.top_products.values[i]
                }))} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
                  <Bar dataKey="value" fill={COLORS[3]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfica 5: Cantidad por Categoría */}
          {chartData.quantity_by_category && (
            <div className="chart-card" data-testid="chart-quantity-by-category">
              <h3 className="chart-title">{chartData.quantity_by_category.title}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData.quantity_by_category.labels.map((label, i) => ({
                      name: label,
                      value: chartData.quantity_by_category.values[i]
                    }))}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => `${entry.name}: ${entry.value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chartData.quantity_by_category.labels.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfica 6: Ganancia por Categoría */}
          {chartData.profit_by_category && (
            <div className="chart-card" data-testid="chart-profit-by-category">
              <h3 className="chart-title">{chartData.profit_by_category.title}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData.profit_by_category.labels.map((label, i) => ({
                  name: label,
                  value: chartData.profit_by_category.values[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
                  <Bar dataKey="value" fill={COLORS[5]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Estadísticas Descriptivas Detalladas */}
      <div className="section-card">
        <h2 className="section-title" data-testid="descriptive-stats-title">📊 Análisis Estadístico Descriptivo</h2>
        
        <div className="stats-detail-grid">
          {/* Precio Total */}
          <div className="stats-detail-card" data-testid="stats-precio-total">
            <h4>💰 Precio Total</h4>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Media:</span>
              <span className="stats-detail-value">${stats.precio_total.media.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Mediana:</span>
              <span className="stats-detail-value">${stats.precio_total.mediana.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Moda:</span>
              <span className="stats-detail-value">${stats.precio_total.moda.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Desv. Estándar:</span>
              <span className="stats-detail-value">${stats.precio_total.desviacion_estandar.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Mínimo:</span>
              <span className="stats-detail-value">${stats.precio_total.minimo.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Máximo:</span>
              <span className="stats-detail-value">${stats.precio_total.maximo.toFixed(2)}</span>
            </div>
          </div>

          {/* Cantidad */}
          <div className="stats-detail-card" data-testid="stats-cantidad">
            <h4>📦 Cantidad</h4>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Media:</span>
              <span className="stats-detail-value">{stats.cantidad.media.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Mediana:</span>
              <span className="stats-detail-value">{stats.cantidad.mediana.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Moda:</span>
              <span className="stats-detail-value">{stats.cantidad.moda.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Desv. Estándar:</span>
              <span className="stats-detail-value">{stats.cantidad.desviacion_estandar.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Mínimo:</span>
              <span className="stats-detail-value">{stats.cantidad.minimo}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Máximo:</span>
              <span className="stats-detail-value">{stats.cantidad.maximo}</span>
            </div>
          </div>

          {/* Precio Unitario */}
          <div className="stats-detail-card" data-testid="stats-precio-unitario">
            <h4>🏷️ Precio Unitario</h4>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Media:</span>
              <span className="stats-detail-value">${stats.precio_unitario.media.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Mediana:</span>
              <span className="stats-detail-value">${stats.precio_unitario.mediana.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Desv. Estándar:</span>
              <span className="stats-detail-value">${stats.precio_unitario.desviacion_estandar.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Mínimo:</span>
              <span className="stats-detail-value">${stats.precio_unitario.minimo.toFixed(2)}</span>
            </div>
            <div className="stats-detail-item">
              <span className="stats-detail-label">Máximo:</span>
              <span className="stats-detail-value">${stats.precio_unitario.maximo.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Análisis Predictivo */}
      {predictions && (
        <div className="section-card prediction-section">
          <h2 className="section-title" data-testid="predictive-analysis-title">🔮 Análisis Predictivo - Modelo de Regresión Lineal</h2>
          <p style={{marginBottom: '1.5rem', color: '#666'}}>
            Modelo entrenado para predecir el Precio Total basado en: Cantidad, Precio Unitario y Categoría
          </p>

          {/* Métricas del Modelo */}
          <div className="metrics-grid">
            <div className="metric-card" data-testid="metric-r2">
              <h5>R² Score</h5>
              <p>{predictions.metrics.r2_score.toFixed(4)}</p>
            </div>
            <div className="metric-card" data-testid="metric-rmse">
              <h5>RMSE</h5>
              <p>${predictions.metrics.rmse.toFixed(2)}</p>
            </div>
            <div className="metric-card" data-testid="metric-mae">
              <h5>MAE</h5>
              <p>${predictions.metrics.mae.toFixed(2)}</p>
            </div>
            <div className="metric-card" data-testid="metric-mse">
              <h5>MSE</h5>
              <p>{predictions.metrics.mse.toFixed(2)}</p>
            </div>
          </div>

          {/* Gráfica de Predicciones */}
          <div style={{marginTop: '2rem'}}>
            <h3 style={{marginBottom: '1rem', textAlign: 'center'}}>Valores Reales vs Predichos</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={predictions.predictions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" name="Índice" />
                <YAxis name="Precio" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Valores Reales" data={predictions.predictions} fill={COLORS[0]} dataKey="actual" />
                <Scatter name="Valores Predichos" data={predictions.predictions} fill={COLORS[1]} dataKey="predicted" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Importancia de Características */}
          <div className="feature-importance">
            <h3 style={{marginBottom: '1.5rem'}}>Coeficientes del Modelo (Importancia de Características)</h3>
            {Object.entries(predictions.feature_importance).map(([feature, value]) => {
              const maxValue = Math.max(...Object.values(predictions.feature_importance).map(Math.abs));
              const percentage = (Math.abs(value) / maxValue) * 100;
              
              return (
                <div key={feature} className="importance-item" data-testid={`importance-${feature.toLowerCase()}`}>
                  <span className="importance-label">{feature}</span>
                  <div className="importance-bar-container">
                    <div className="importance-bar" style={{width: `${percentage}%`}}>
                      {value.toFixed(2)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;