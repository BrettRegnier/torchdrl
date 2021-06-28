import matplotlib.pyplot as plt 
import torch    

fil = torch.load("models/checkpoints/CartPole_v4/final/episode_243_score_195.08.pt")
chart_metrics = fil['chart_metrics']

print(fil.keys())
print(chart_metrics.keys())
plt.plot(chart_metrics['episode'], chart_metrics['score'], 'green')
plt.plot(chart_metrics['episode'], chart_metrics['steps'], 'purple')
plt.plot(chart_metrics['episode'], chart_metrics['loss'], 'blue')
plt.show()