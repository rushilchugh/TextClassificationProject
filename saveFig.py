def save(plList, saveName):
	plt.style.use('ggplot')
	plt.bar(*plList)
	plt.title(saveName.split('.')[0])
	plt.xticks(rotation = 70)
	plt.tight_layout()
	plt.savefig(saveName)
	plt.clf()
	plt.close()
