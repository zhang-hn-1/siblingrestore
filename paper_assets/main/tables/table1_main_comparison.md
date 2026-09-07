# table1_main_comparison

Main restoration and identity comparison on the frozen test set. Bold/underline should be applied only by actual column rank; A5 is not the best LPIPS row.

| group | method | PSNR | SSIM | LPIPS | Top1 | EER |
|---|---|---|---|---|---|---|
| Ours | Ours | 28.790507863360915 | 0.9457700604481423 | 0.1433372255277001 | 0.6981026785714286 | 0.07988773634453782 |
| General Restoration | Restormer | 28.104552100132697 | 0.943308229890785 | 0.14961439062972204 | 0.65625 | 0.09017528886554621 |
| All-in-One Restoration | DehazeFormer | 28.37168088442346 | 0.9433459896328193 | 0.1375604359600402 | 0.6707589285714286 | 0.09418658088235293 |
| All-in-One Restoration | PromptIR | 27.525529067921436 | 0.938064118498005 | 0.15789028613549558 | 0.6333705357142857 | 0.09438681722689077 |
