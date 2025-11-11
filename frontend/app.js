const trainBtn = document.getElementById('trainBtn');
const loadBtn = document.getElementById('loadBtn');
const statusEl = document.getElementById('status');
const rnnMaeEl = document.getElementById('rnnMae');
const rnnRmseEl = document.getElementById('rnnRmse');
const lstmMaeEl = document.getElementById('lstmMae');
const lstmRmseEl = document.getElementById('lstmRmse');
const ctx = document.getElementById('predChart').getContext('2d');

let chart;

function setStatus(text) {
	statusEl.textContent = text;
}

function renderChart(actual, lstm, rnn) {
	const labels = Array.from({ length: actual.length }, (_, i) => i + 1);
	const data = {
		labels,
		datasets: [
			{
				label: 'Actual',
				data: actual,
				borderColor: '#22d3ee',
				backgroundColor: 'rgba(34,211,238,0.15)',
				tension: 0.25,
				borderWidth: 2
			},
			{
				label: 'LSTM Predicted',
				data: lstm,
				borderColor: '#a78bfa',
				backgroundColor: 'rgba(167,139,250,0.15)',
				tension: 0.25,
				borderDash: [6, 4],
				borderWidth: 2
			},
			{
				label: 'RNN Predicted',
				data: rnn,
				borderColor: '#fca5a5',
				backgroundColor: 'rgba(252,165,165,0.10)',
				tension: 0.25,
				borderDash: [2, 3],
				borderWidth: 2
			}
		]
	};
	const options = {
		responsive: true,
		plugins: {
			legend: { labels: { color: '#cbd5e1' } },
			tooltip: { mode: 'index', intersect: false }
		},
		scales: {
			x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
			y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
		},
		interaction: { mode: 'index', intersect: false }
	};

	if (chart) {
		chart.data = data;
		chart.options = options;
		chart.update();
	} else {
		chart = new Chart(ctx, { type: 'line', data, options });
	}
}

async function runTraining() {
	try {
		setStatus('Training (this may take ~10-60s)...');
		trainBtn.disabled = true;
		const res = await fetch('/api/train');
		const json = await res.json();
		if (!res.ok) {
			throw new Error(json.error || 'Request failed');
		}
		const { metrics, series } = json;
		rnnMaeEl.textContent = metrics.rnn.mae.toFixed(4);
		rnnRmseEl.textContent = metrics.rnn.rmse.toFixed(4);
		lstmMaeEl.textContent = metrics.lstm.mae.toFixed(4);
		lstmRmseEl.textContent = metrics.lstm.rmse.toFixed(4);
		renderChart(series.actual, series.lstm, series.rnn);
		setStatus('Done');
	} catch (e) {
		console.error(e);
		setStatus('Error: ' + e.message);
		alert('Error: ' + e.message);
	} finally {
		trainBtn.disabled = false;
	}
}

trainBtn.addEventListener('click', runTraining);

async function loadPrecomputed() {
	try {
		setStatus('Loading frontend/data.json...');
		loadBtn.disabled = true;
		const res = await fetch('data.json', { cache: 'no-store' });
		const json = await res.json();
		if (!res.ok) {
			throw new Error(json.error || 'Failed to load data.json');
		}
		const { metrics, series } = json;
		rnnMaeEl.textContent = metrics.rnn.mae.toFixed(4);
		rnnRmseEl.textContent = metrics.rnn.rmse.toFixed(4);
		lstmMaeEl.textContent = metrics.lstm.mae.toFixed(4);
		lstmRmseEl.textContent = metrics.lstm.rmse.toFixed(4);
		renderChart(series.actual, series.lstm, series.rnn);
		setStatus('Done');
	} catch (e) {
		console.error(e);
		setStatus('Error: ' + e.message);
		alert('Make sure frontend/data.json exists and you are serving via http (python -m http.server). Error: ' + e.message);
	} finally {
		loadBtn.disabled = false;
	}
}

loadBtn.addEventListener('click', loadPrecomputed);


