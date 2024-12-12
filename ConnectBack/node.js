const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
app.use(bodyParser.json());

// Python 추론 API 호출
app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post('http://localhost:5000/predict', req.body);
        res.json(response.data); // Python API의 결과를 클라이언트로 반환
    } catch (error) {
        console.error('Error calling Python API:', error.message);
        res.status(500).json({ error: 'Python API 호출 중 문제가 발생했습니다.' });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Node.js 서버가 http://localhost:${PORT} 에서 실행 중입니다.`);
});
