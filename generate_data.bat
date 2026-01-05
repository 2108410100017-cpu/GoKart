@echo off
echo Generating 20 prediction requests to build up live data...

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":28,\"time_of_day\":12}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":32,\"time_of_day\":15}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":25,\"time_of_day\":10}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":35,\"time_of_day\":14}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":20,\"time_of_day\":18}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":31,\"time_of_day\":13}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":29,\"time_of_day\":16}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":33,\"time_of_day\":11}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":27,\"time_of_day\":17}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":34,\"time_of_day\":14}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":22,\"time_of_day\":9}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":30,\"time_of_day\":15}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":26,\"time_of_day\":19}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":36,\"time_of_day\":13}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":24,\"time_of_day\":8}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":31,\"time_of_day\":16}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":29,\"time_of_day\":12}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":37,\"time_of_day\":15}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":23,\"time_of_day\":20}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"temperature\":30,\"time_of_day\":14}"

echo.
echo Done! 20 requests sent. Now run the cycle.
pause